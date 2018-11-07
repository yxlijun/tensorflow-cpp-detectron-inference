#ifndef DETECTORSDD_HPP_
#define DETECTORSDD_HPP_


#include "Detector.hpp"
#include "utils.hpp"

struct objectinfo {
	cv::Rect bbox;
	int classes;
	float score;
};


class Detectorsdd: public Detector
{
private:
	vector<string> _out_tensor_names;
	vector<int> _feat_shapes;
	vector<vector<float>> _anchor_sizes;
	vector<vector<float>> _anchor_ratios;
	vector<int> _anchor_steps;
	vector<int> img_shape;
	vector<float> prior_scaling;
	vector<int> _anchor_nums;
	vector<vector<float>> get_anchors(int index);
	void bboxes_sort(vector<vector<float>> &bboxes,
	                 vector<float> &scores,
	                 vector<int> &classes,
	                 vector<vector<float>> &_bboxes,
	                 vector<float> &_scores,
	                 vector<int> &_classes, int topk = 40);

	vector<cv::Rect> box_to_corner(vector<vector<float>> bboxes,
	                               int width,
	                               int height);

public:
	Detectorsdd(const string &model_path);

	virtual ~Detectorsdd();

	virtual void detect(const string &image_path);

	virtual void draw_bbox_image();

private:
	Session *session;
	string _image_path;
	float _score_thresh;
	float _nms_thresh;

	vector<string> _voc_classes;
	vector<objectinfo> _objectinfo;

};
#endif