#include "yolo_common.hpp"

#include <opencv2/imgproc.hpp>
#include "im2d.h"
#include "preprocess.h"
#include "yolov8/postprocess_v8.h"

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
    unsigned char *data;
    int ret;

    data = NULL;

    if (NULL == fp)
    {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0)
    {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char *)malloc(sz);
    if (data == NULL)
    {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp;
    unsigned char *data;

    fp = fopen(filename, "rb");
    if (NULL == fp)
    {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    std::string shape_str = attr->n_dims < 1 ? "" : std::to_string(attr->dims[0]);
    for (int i = 1; i < attr->n_dims; ++i)
    {
        shape_str += ", " + std::to_string(attr->dims[i]);
    }
}


YoloModel::YoloModel(std::string modelPath, int num_classes_, ModelVersion type_, int coreNumber)
    : numClasses(num_classes_), yoloType(type_) {

    printf("Loading model...\n");
    int model_data_size = 0;
    model_data = load_model(modelPath.c_str(), &model_data_size);

    // 模型参数复用/Model parameter reuse
    int ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);

    if (ret < 0)
    {
        throw std::runtime_error("rknn_init error ret=" + ret);
    }

    // hard coded to let npu decide where the model runs
    this->changeCoreMask(coreNumber);

    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0)
    {
        throw std::runtime_error("rknn_init error ret=" + ret);
    }
    printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);

    // 获取模型输入输出参数/Obtain the input and output parameters of the model
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        throw std::runtime_error("rknn_init error ret=" + ret);
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // 设置输入参数/Set the input parameters

    // input_attrs = (rknn_tensor_attr *)calloc(io_num.n_input, sizeof(rknn_tensor_attr));
    input_attrs.resize(io_num.n_input);

    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            throw std::runtime_error("rknn_init error ret=" + ret);
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // 设置输出参数/Set the output parameters

    // output_attrs = (rknn_tensor_attr *)calloc(io_num.n_output, sizeof(rknn_tensor_attr));
    output_attrs.resize(io_num.n_output);

    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        dump_tensor_attr(&(output_attrs[i]));
    }

    // TODO
    if (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && output_attrs[0].type == RKNN_TENSOR_INT8)
    {
        is_quant = true;
    }
    else
    {
        is_quant = false;
    }
    printf("Model quantized? %s\n", is_quant ? "YES" : "NO");

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        printf("model is NCHW input fmt\n");
        channels = input_attrs[0].dims[1];
        int height = input_attrs[0].dims[2];
        int width = input_attrs[0].dims[3];
        modelSize = {width, height};
    }
    else
    {
        printf("model is NHWC input fmt\n");
        int height = input_attrs[0].dims[1];
        int width = input_attrs[0].dims[2];
        channels = input_attrs[0].dims[3];
        modelSize = {width, height};
    }
    printf("model input height=%d, width=%d, channel=%d\n", 
        modelSize.width, modelSize.height, channels);

    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = modelSize.width * modelSize.height * channels;
    inputs[0].fmt = RKNN_TENSOR_NHWC; // TODO matt: this seems smelly
    inputs[0].pass_through = 0;
}


YoloModel::~YoloModel() {
    rknn_destroy(ctx);

    if (model_data)
        free(model_data);
}

int YoloModel::changeCoreMask(int coreNumber) {
        // 设置模型绑定的核心/Set the core of the model that needs to be bound
    rknn_core_mask core_mask;
    switch (coreNumber)
    {
    case -1:
        core_mask = RKNN_NPU_CORE_AUTO;
        break;
    case 0:
        core_mask = RKNN_NPU_CORE_0;
        break;
    case 1:
        core_mask = RKNN_NPU_CORE_1;
        break;
    case 2:
        core_mask = RKNN_NPU_CORE_2;
        break;
    case 10:
        core_mask = RKNN_NPU_CORE_0_1;
        break;
    case 210:
        core_mask = RKNN_NPU_CORE_0_1_2;
        break;
    default:
        throw std::runtime_error("invalid core selection! core selected: " + coreNumber);
        break;
    }
    int ret = rknn_set_core_mask(ctx, core_mask);
    if (ret < 0)
    {
        throw std::runtime_error("rknn_init core error ret=" + ret);
    }
    return ret;
}

detect_result_group_t YoloModel::forward(cv::Mat &orig_img, DetectionFilterParams params) {
    cv::Mat img;
    cv::cvtColor(orig_img, img, cv::COLOR_BGR2RGB);
    int img_width = img.cols;
    int img_height = img.rows;

    BOX_RECT pads;
    memset(&pads, 0, sizeof(BOX_RECT));
    cv::Mat resized_img(modelSize, CV_8UC3);
    // 计算缩放比例/Calculate the scaling ratio
    float scale_w = (float)modelSize.width / img.cols;
    float scale_h = (float)modelSize.height / img.rows;

    // 图像缩放/Image scaling
    if (img_width != modelSize.width || img_height != modelSize.height)
    {
        // rga
        rga_buffer_t src;
        rga_buffer_t dst;
        memset(&src, 0, sizeof(src));
        memset(&dst, 0, sizeof(dst));
        int ret = resize_rga(src, dst, img, resized_img, modelSize);
        if (ret != 0)
        {
            fprintf(stderr, "resRGA_CURRENT_API_HEADER_VERSIONize with rga error\n");
        }
        /*********
        // opencv
        float min_scale = std::min(scale_w, scale_h);
        scale_w = min_scale;
        scale_h = min_scale;
        letterbox(img, resized_img, pads, min_scale, target_size);
        *********/
        inputs[0].buf = resized_img.data;
    }
    else
    {
        inputs[0].buf = img.data;
    }

    rknn_inputs_set(ctx, io_num.n_input, inputs);

    std::vector<rknn_output> outputs;
    outputs.resize(io_num.n_output);

    for (int i = 0; i < io_num.n_output; i++)
    {
        memset(&outputs[i], 0, sizeof(rknn_output));
        outputs[i].want_float = !is_quant;
    }

    // 模型推理/Model inference
    int ret = rknn_run(ctx, NULL);
    ret |= rknn_outputs_get(ctx, io_num.n_output, outputs.data(), NULL);

    if (ret) {
        // todo barf
    }

    // todo box-rect things
    auto detections = this->postProcess(
        outputs, params, orig_img.size(), {scale_w, scale_h}, BOX_RECT{}
    );

    ret = rknn_outputs_release(ctx, io_num.n_output, outputs.data());
    if (ret) {
        // todo barf
    }

    return detections;
}

detect_result_group_t YoloV5Model::postProcess(std::vector<rknn_output> outputs,
        DetectionFilterParams params, 
        cv::Size inputImageSize,
        cv::Size2d imageScale,
        BOX_RECT letterbox
    ) {
    detect_result_group_t result;

    BOX_RECT pads;
    memset(&pads, 0, sizeof(BOX_RECT));

    // 后处理/Post-processing
    std::vector<float> out_scales;
    std::vector<int32_t> out_zps;
    for (int i = 0; i < io_num.n_output; ++i)
    {
        out_scales.push_back(output_attrs[i].scale);
        out_zps.push_back(output_attrs[i].zp);
    }

    post_process_v5((int8_t *)outputs[0].buf, (int8_t *)outputs[1].buf, (int8_t *)outputs[2].buf, 
                modelSize.width, modelSize.height,
                params.box_thresh, params.nms_thresh, pads, 
                imageScale.width, imageScale.height, out_zps, out_scales, &result, numClasses);

    return result;
}


detect_result_group_t YoloV8Model::postProcess(std::vector<rknn_output> outputs,
        DetectionFilterParams params, 
        cv::Size inputImageSize,
        cv::Size2d imageScale,
        BOX_RECT letterbox) {
    detect_result_group_t result;

    BOX_RECT padding {
        0,
        inputImageSize.width,
        0,
        inputImageSize.height,
    };

    post_process_v8(modelSize, outputs.data(), &padding, params.box_thresh, params.nms_thresh, &result, 
        numClasses, output_attrs, is_quant, io_num.n_output);

    return result;
}
