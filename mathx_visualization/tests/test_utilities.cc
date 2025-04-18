#include "test_utilities.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace mathx {
namespace visualization {

cv::Mat GenerateTestImage(const int width, const int height) {
  cv::Mat image(height, width, CV_8UC3);
  image = 0;
  cv::circle(image, cv::Point(width / 2, height / 2), width / 5,
             cv::Scalar(255, 255, 255));

  cv::putText(image, "red", cv::Point(width / 20, height / 20),
              cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255));
  cv::putText(image, "green", cv::Point(width / 20, height / 20 * 2),
              cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0));
  cv::putText(image, "blue", cv::Point(width / 20, height / 20 * 3),
              cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0));

  // cv::imshow("GenerateImage", image);
  // cv::waitKey();

  return image;
}

}  // namespace visualization
}  // namespace mathx
