#ifndef DECTORYOLOV2_HPP_
#define DECTORYOLOV2_HPP_

#include "Detector.hpp"
#include "utils.hpp"
#include <map>
#include <fstream>

struct OjbectInfo {
	cv::Rect bbox;
	int classes;
	float score;
};

struct Config {
	Config();
	int width, height;
	float mean_r, mean_g, mean_b;
	std::map<int, string> VOC_CLASSES;
};


class Detectoryolov1: public Detector
{
public:
	Detectoryolov1(const string &model_path, float thresh = 0.1);

	virtual ~Detectoryolov1();

	virtual void detect(const string &image_path);

	virtual void draw_bbox_image();

	cv::Mat preprocess(const string &image_path);

	vector<OjbectInfo> get_objectInfo();
private:
	Session* session;
	string _image_path;
	Config _config;
	float _thresh;

	vector<OjbectInfo> _objectinfo;
};
#endif