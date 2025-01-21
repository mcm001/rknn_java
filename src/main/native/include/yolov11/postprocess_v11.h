#ifndef _RKNN_YOLOV11_DEMO_POSTPROCESS_H_
#define _RKNN_YOLOV11_DEMO_POSTPROCESS_H_

#include <stdint.h>
#include <vector>
#include "rknn_api.h"
#include "common.h"
#include <opencv2/core/types.hpp>
#include "yolov5/postprocess_v5.h"

#define OBJ_NAME_MAX_SIZE_V11 64
#define OBJ_NUMB_MAX_SIZE_V11 128
// #define OBJ_CLASS_NUM 3
// #define NMS_THRESH 0.45
// #define BOX_THRESH 0.25


// int init_post_process();
// void deinit_post_process();
// char *coco_cls_to_name(int cls_id);
int post_process_v11(
    cv::Size modelSize,
    rknn_output *outputs, BOX_RECT *letter_box, float conf_threshold, float nms_threshold, detect_result_group_t *od_results, int numClasses, std::vector<rknn_tensor_attr> &output_attrs, bool is_quant, int n_outputs);

#endif //_RKNN_YOLOV11_DEMO_POSTPROCESS_H_
