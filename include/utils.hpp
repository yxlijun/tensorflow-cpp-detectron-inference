#ifndef UTILS_HPP
#define UTILS_HPP

#include "Detector.hpp"

float iou(const cv::Rect& r1, const cv::Rect &r2);

vector<int> nms(vector<cv::Rect> & proposal, vector<float> scores, const float thresh = 0.5);

string& trim(string &str);


vector<int> argsort(vector<float> scores);
#endif