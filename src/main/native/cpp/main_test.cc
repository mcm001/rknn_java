#include "yolo_common.hpp"
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

int main_test(ModelVersion version)
{

    YoloModel *wrapper;

    if (version == ModelVersion::YOLO_V5) {
        printf("Starting with version 5\n");
        wrapper = new YoloV5Model("note-640-640-yolov5s.rknn", 1, 0);
    } else if (version == ModelVersion::YOLO_V8) {
        printf("Starting with version 8\n");
        wrapper = new YoloV8Model("note-robot-yolov8s-quant.rknn", 3, 0);
    } else if (version == ModelVersion::YOLO_V11){
        printf("Starting with version 11\n");
        wrapper = new YoloV11Model("note-robot-yolov11s-quant.rknn", 3, 0);
    } else {
        printf("Unknown version\n");
        return 1;
    }
    
    std::cout << "created: " << (long unsigned int)wrapper << std::endl;

    for (int j = 0; j < 1; j++) {
        cv::Mat img;
        img = cv::imread("robots.png");

        DetectionFilterParams params {
            .nms_thresh = 0.45,
            .box_thresh = 0.15,
        };
        auto ret = wrapper->forward(img, params);

        std::cout << "Count: " << ret.count << std::endl;
        for (int i = 0; i < ret.count; ++i) {
            std::cout << "ID: " << ret.results[i].id << " conf " << ret.results[i].obj_conf << " @ "
                << ret.results[i].box.top << " - "
                << ret.results[i].box.left << " - "
                << ret.results[i].box.bottom << " - "
                << ret.results[i].box.right << " - "
                << std::endl;

            auto *det_result = &(ret.results[i]);

            int x1 = det_result->box.left;
            int y1 = det_result->box.top;
            int x2 = det_result->box.right;
            int y2 = det_result->box.bottom;

            cv::rectangle(
                img, cv::Rect(x1, y1, x2-x1, y2-y1), (0,255,0), 3 
            );

            // draw_rectangle(&src_image, x1, y1, x2 - x1, y2 - y1, COLOR_BLUE, 3);
            char text[256];
            sprintf(text, "id%i %.1f%%", det_result->id, det_result->obj_conf * 100);
            cv::putText(img, text, {x1, y1 + 10}, cv::FONT_ITALIC, 0.6, {100, 255, 0});
        }

        cv::imwrite("out2.png", img);
    }

    std::cout << "Deleting: " << (long unsigned int)wrapper << std::endl;
    delete wrapper;

    return 0;
}

#include <thread>
int main() {
//     std::vector<std::thread> threads;
// for (int i=1; i<=1; ++i)
//     threads.emplace_back(std::thread([]() {main_test("../note-640-640-yolov5s.rknn");}));
//     for (auto& th : threads) th.join();

    main_test(ModelVersion::YOLO_V8);
    // main_test(ModelVersion::YOLO_V5);
}
