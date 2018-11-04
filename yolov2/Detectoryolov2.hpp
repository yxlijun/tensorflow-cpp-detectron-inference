#ifndef DETECTORYOLOV2_HPP_
#define	DETECTORYOLOV2_HPP_

#include "Detector.hpp"
#include "utils.hpp"


struct objectinfo {
	cv::Rect bbox;
	int classes;
	float score;
};

class Detectoryolov2: public Detector
{
private:
	float sigmoid(float x);

	void sotfmax(vector<float> &vals);

	void box_to_corner(vector<vector<float>> &boxes);

	void filter_box(vector<vector<float>> &boxes,
	                vector<vector<float>> &fin_boxes,
	                vector<float> &box_conf,
	                vector<float> &scores,
	                vector<vector<float>> &classes_conf,
	                vector<float> &classes_prob);

public:
	Detectoryolov2(const string &model_path);

	virtual ~Detectoryolov2();

	virtual void detect(const string &image_path);

	virtual void draw_bbox_image();

	cv::Mat preprocess(const string &image_path);

	void set_nms_thresh(float thresh);

	void set_score_thresh(float thresh);


private:
	Session *session;
	string _image_path;
	const int _inputWidth;
	const int _inputHeight;
	const int _featsize;
	const int NUM_BOXES_PER_BLOCK;
	const int _classes_num;
	vector<vector<float>> _anchors;
	vector<string> _voc_classes;

	float score_thresh;
	float nms_thresh;
	int ori_width;
	int ori_height;
	vector<objectinfo> _objectinfo;

};

#endif