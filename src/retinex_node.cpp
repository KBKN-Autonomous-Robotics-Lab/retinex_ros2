#include <opencv2/opencv.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>

int main(int argc, char * argv[]){
  rclcpp::init(argc, argv);

  auto node = rclcpp::Node::make_shared("retinex_node");
  auto image_pub = node->create_publisher<sensor_msgs::msg::Image>("result_image", rclcpp::QoS(10));

  rclcpp::WallRate loop_rate(1);

  cv::VideoCapture cap(4);
  cap.set(cv::CAP_PROP_FRAME_WIDTH, 960);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 544);
  /*
  '320x240'
  '352x288'
  '640x360'
  '640x480'
  '752x416'
  '800x600'
  '960x544'
  '960x720'
  '1024x576' default
  */
  cv::Mat frame,image_xyz, image_rgb, im1, im2, im3, im4, im5;
  cv::Mat image_gauss, image_gauss_xyz;
  cv::Mat rgb_gauss_channels[3];
  cv::Mat xyz_gauss_channels[3];
  cv::Mat R_channels[3];
  cv::Mat SE = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(21, 21));

  while(rclcpp::ok()){
    cap >> frame;
    if(frame.empty()) break;

    cv::cvtColor(frame, image_rgb, cv::COLOR_BGR2RGB); 
    cv::cvtColor(image_rgb, image_xyz, cv::COLOR_RGB2XYZ); 
    cv::GaussianBlur(image_rgb, image_gauss, cv::Size(41, 41), 8); 
    cv::cvtColor(image_gauss, image_gauss_xyz, cv::COLOR_RGB2XYZ);

    cv::split(image_gauss, rgb_gauss_channels); 
    cv::Mat r_gauss = rgb_gauss_channels[0];
    cv::Mat g_gauss = rgb_gauss_channels[1];
    cv::Mat b_gauss = rgb_gauss_channels[2];
    cv::Mat Luminance = 0.298912*r_gauss+0.586611*g_gauss+0.114478*b_gauss;

    double minVal, maxVal;
    cv::minMaxLoc(Luminance, &minVal);
    cv::minMaxLoc(Luminance, nullptr, &maxVal);
    cv::Mat Luminance_Ratio = (Luminance-minVal)/(maxVal-minVal);

    cv::split(image_gauss_xyz, xyz_gauss_channels);
    cv::Mat x_gauss = xyz_gauss_channels[0];
    cv::Mat y_gauss = xyz_gauss_channels[1];
    cv::Mat z_gauss = xyz_gauss_channels[2];

    cv::Mat XYZ_Color = x_gauss+y_gauss+z_gauss;
    cv::Mat XYZ_Color_mask = (XYZ_Color == 0); 
    XYZ_Color.setTo(cv::Scalar(1e-6), XYZ_Color_mask);
    cv::Mat XYZ_Color_X, XYZ_Color_Y, R, R_log;
    cv::divide(x_gauss, XYZ_Color, XYZ_Color_X);
    cv::divide(y_gauss, XYZ_Color, XYZ_Color_Y);
    cv::Mat Luminance_3ch;
    cv::Mat Luminance_channels[] = { Luminance, Luminance, Luminance };
    cv::merge(Luminance_channels, 3, Luminance_3ch);
    cv::divide(image_rgb, Luminance_3ch, R_log);
    cv::Mat R_log_float;
    R_log.convertTo(R_log_float, CV_32F);
    cv::log(R_log_float + 1, R);

    cv::split(R, R_channels);
    cv::Mat R_r = R_channels[0];
    cv::Mat R_g = R_channels[1];
    cv::Mat R_b = R_channels[2];

    cv::Mat cond1, cond2, cond3, cond4, cond5, cond6, cond7;
    cv::Mat I_xyz_red;
    cv::compare(XYZ_Color_Y, -0.052631578947368*XYZ_Color_X + 0.34631578947368, cond1, cv::CMP_GE);
    cv::compare(XYZ_Color_Y, -1*XYZ_Color_X + 0.91, cond2, cv::CMP_GE);
    cv::compare(XYZ_Color_Y, 0.058823529411765*XYZ_Color_X + 0.30647058823529, cond3, cv::CMP_LE);
    cv::compare(Luminance_Ratio, 0.25, cond4, cv::CMP_GE);
    cv::compare(R_r, 0.1, cond5, cv::CMP_GE);
    cv::compare(0, R_b, cond6, cv::CMP_GE);
    cv::bitwise_and(cond1, cond2, I_xyz_red);
    cv::bitwise_and(I_xyz_red, cond3, I_xyz_red);
    cv::bitwise_and(I_xyz_red, cond4, I_xyz_red);
    cv::bitwise_and(I_xyz_red, cond5, I_xyz_red);
    cv::bitwise_and(I_xyz_red, cond6, I_xyz_red);

    cv::Mat I_xyz_orange;
    cv::compare(XYZ_Color_Y, 0.2*XYZ_Color_X + 0.268, cond1, cv::CMP_LE);
    cv::compare(XYZ_Color_Y, -1*XYZ_Color_X + 0.91, cond2, cv::CMP_GE);
    cv::compare(XYZ_Color_Y, 0.058823529411765*XYZ_Color_X + 0.30647058823529, cond3, cv::CMP_GE);
    cv::compare(Luminance_Ratio, 0.4, cond4, cv::CMP_GE);
    cv::compare(R_r, 0, cond5, cv::CMP_GE);
    cv::compare(0, R_b, cond6, cv::CMP_GE);
    cv::bitwise_and(cond1, cond2, I_xyz_orange);
    cv::bitwise_and(I_xyz_red, cond3, I_xyz_orange);
    cv::bitwise_and(I_xyz_red, cond4, I_xyz_orange);
    cv::bitwise_and(I_xyz_red, cond5, I_xyz_orange);
    cv::bitwise_and(I_xyz_red, cond6, I_xyz_orange);

    cv::Mat I_xyz_yellow;
    cv::compare(XYZ_Color_Y, 3.741935483871*XYZ_Color_X - 0.83812903225807, cond1, cv::CMP_LE);
    cv::compare(XYZ_Color_Y, -1*XYZ_Color_X + 0.85, cond2, cv::CMP_GE);
    cv::compare(XYZ_Color_Y, 1.4193548387097*XYZ_Color_X - 0.11290322580645, cond3, cv::CMP_GE);
    cv::compare(Luminance_Ratio, 0.7, cond4, cv::CMP_GE);
    cv::compare(R_r, 0.1, cond5, cv::CMP_GE);
    cv::compare(R_g, 0.1, cond6, cv::CMP_GE);
    cv::compare(0, R_b, cond7, cv::CMP_GE);
    cv::bitwise_and(cond1, cond2, I_xyz_yellow);
    cv::bitwise_and(I_xyz_red, cond3, I_xyz_yellow);
    cv::bitwise_and(I_xyz_red, cond4, I_xyz_yellow);
    cv::bitwise_and(I_xyz_red, cond5, I_xyz_yellow);
    cv::bitwise_and(I_xyz_red, cond6, I_xyz_yellow);
    cv::bitwise_and(I_xyz_red, cond7, I_xyz_yellow);

    cv::morphologyEx(I_xyz_red, I_xyz_red, cv::MORPH_CLOSE, SE);
    cv::morphologyEx(I_xyz_orange, I_xyz_orange, cv::MORPH_CLOSE, SE);
    cv::morphologyEx(I_xyz_yellow, I_xyz_yellow, cv::MORPH_CLOSE, SE);

    std::vector<std::vector<cv::Point>> contours_red, contours_orange, contours_yellow;
    cv::findContours(I_xyz_red, contours_red, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::findContours(I_xyz_orange, contours_orange, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::findContours(I_xyz_yellow, contours_yellow, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<std::vector<cv::Point>> largeContours_red, largeContours_orange, largeContours_yellow;
    for (const auto& contour : contours_red) {
      if (cv::contourArea(contour) > 100)
        largeContours_red.push_back(contour);
    }
    for (const auto& contour : contours_orange) {
      if (cv::contourArea(contour) > 100)
        largeContours_orange.push_back(contour);
    }
    for (const auto& contour : contours_yellow) {
      if (cv::contourArea(contour) > 100)
        largeContours_yellow.push_back(contour);
    }
    cv::Mat output_red = cv::Mat::zeros(I_xyz_red.size(), CV_8U);
    cv::Mat output_orange = cv::Mat::zeros(I_xyz_orange.size(), CV_8U);
    cv::Mat output_yellow = cv::Mat::zeros(I_xyz_yellow.size(), CV_8U);

    cv::drawContours(output_red, largeContours_red, -1, cv::Scalar(255), cv::FILLED);
    cv::drawContours(output_orange, largeContours_orange, -1, cv::Scalar(255), cv::FILLED);
    cv::drawContours(output_yellow, largeContours_yellow, -1, cv::Scalar(255), cv::FILLED);

    cv::Mat composite_result = I_xyz_red | I_xyz_orange | I_xyz_yellow;
    sensor_msgs::msg::Image::SharedPtr msg = cv_bridge::CvImage(std_msgs::msg::Header(), "mono8", composite_result).toImageMsg();
    image_pub->publish(*msg);
  }
  
  rclcpp::spin_some(node);
  rclcpp::shutdown();

  return 0;
}
