#include "Detectoryolov2.hpp"


static void image_to_vector(vector<float>& data, const cv::Mat &image_np) {
	int w = image_np.cols, h = image_np.rows;
	for (int i = 0; i < h; ++i) {
		for (int j = 0; j < w; ++j) {
			for (int k = 0; k < 3; ++k) {
				data.push_back(static_cast<float>(image_np.at<cv::Vec3f>(i, j)[k]));
			}
		}
	}
}

Detectoryolov2::Detectoryolov2(const string &model_path): Detector(model_path),
	_inputWidth(416), _inputHeight(416),
	_featsize(13), NUM_BOXES_PER_BLOCK(5),
	score_thresh(0.3), nms_thresh(0.5), _classes_num(20) {
	session = initializer(model_path);
	_anchors = {
		{1.08, 1.19},
		{3.42, 4.41},
		{6.63, 11.38},
		{9.42, 5.11},
		{16.62, 10.52}
	};
	_voc_classes = {
		"aeroplane",
		"bicycle",
		"bird",
		"boat",
		"bottle",
		"bus",
		"car",
		"cat",
		"chair",
		"cow",
		"diningtable",
		"dog",
		"horse",
		"motorbike",
		"person",
		"pottedplant",
		"sheep",
		"sofa",
		"train",
		"tvmonitor"
	};
}

Detectoryolov2::~Detectoryolov2() {
	session->Close();
}

cv::Mat Detectoryolov2::preprocess(const string &image_path) {
	cv::Mat image = cv::imread(image_path);
	ori_height = image.rows, ori_width = image.cols;
	image.convertTo(image, CV_32FC3);
	cv::Mat dstimage;
	cv::resize(image, dstimage, cv::Size(_inputWidth, _inputHeight));
	for (int i = 0; i < _inputHeight; ++i) {
		for (int j = 0; j < _inputWidth; ++j) {
			for (int k = 0; k < 3; ++k) {
				dstimage.at<cv::Vec3f>(i, j)[k] /= 255.0;
			}
		}
	}
	return dstimage;
}


void Detectoryolov2::set_nms_thresh(float thresh) {
	nms_thresh = thresh;
}

void Detectoryolov2::set_score_thresh(float thresh) {
	score_thresh = thresh;
}


void Detectoryolov2::detect(const string &image_path) {
	_image_path = image_path;
	string image_layer = "input";
	string output_layer = "output";

	cv::Mat image = preprocess(image_path);

	vector<float> input_image;
	image_to_vector(input_image, image);
	Tensor input_tensor(tensorflow::DT_FLOAT, TensorShape({1, _inputHeight, _inputWidth, 3}));
	auto input_tensor_mapped = input_tensor.tensor<float, 4>();
	float* out = input_tensor_mapped.data();
	memcpy(out, &input_image[0], input_image.size()*sizeof(float));

	vector<Tensor> outputs;
	vector<std::pair<string, Tensor>> inputs = {{image_layer, input_tensor}};
	vector<string> out_tensor_names = {output_layer};
	Status status = session->Run(inputs, out_tensor_names, {}, &outputs);
	if (!status.ok())
		throw logic_error(status.ToString());
	else
		std::cout << "run model successfully" << std::endl;

	Tensor* detout = &outputs[0];
	auto predicts = detout->shaped<float, 4>({_featsize, _featsize, NUM_BOXES_PER_BLOCK, 25});

	vector<vector<float>> boxes;
	vector<float> box_conf;
	vector<vector<float>> classes_conf;
	for (int i = 0; i < _featsize; ++i) {
		for (int j = 0; j < _featsize; ++j) {
			for (int k = 0; k < NUM_BOXES_PER_BLOCK; ++k) {
				vector<float> box;
				box.push_back(sigmoid(predicts(i, j, k, 0)) + j);
				box.push_back(sigmoid(predicts(i, j, k, 1)) + i);
				box.push_back(std::exp(predicts(i, j, k, 2))*_anchors[k][0]);
				box.push_back(std::exp(predicts(i, j, k, 3))*_anchors[k][1]);

				float conf = predicts(i, j, k, 4);
				box_conf.push_back(sigmoid(conf));

				vector<float> classes;
				for (int c = 5; c < 25; ++c) {
					classes.push_back(predicts(i, j, k, c));
				}
				sotfmax(classes);
				classes_conf.push_back(classes);
				boxes.push_back(box);
			}
		}
	}
	box_to_corner(boxes);
	vector<vector<float>> fin_boxes;
	vector<float> scores;
	vector<float> classes_prob;
	filter_box(boxes, fin_boxes, box_conf, scores, classes_conf, classes_prob);
	const int box_num = fin_boxes.size();
	vector<cv::Rect> proposals(box_num);
	for (int i = 0; i < box_num; ++i) {
		proposals[i].x = static_cast<int>(fin_boxes[i][0] * ori_width);
		proposals[i].y = static_cast<int>(fin_boxes[i][1] * ori_height);
		proposals[i].width = static_cast<int>(fin_boxes[i][2] * ori_width) - proposals[i].x;
		proposals[i].height = static_cast<int>(fin_boxes[i][3] * ori_height) - proposals[i].y;
	}
	vector<int> keep = nms(proposals, scores, nms_thresh);
	const int res_num = keep.size();
	for (int i = 0; i < res_num; ++i) {
		objectinfo _info;
		_info.bbox = proposals[keep[i]];
		_info.classes = classes_prob[keep[i]];
		_info.score = scores[keep[i]];
		_objectinfo.push_back(_info);
	}
}

float Detectoryolov2::sigmoid(float x) {
	return static_cast<float>(1.0) / (1 + std::exp(-x));
}

void Detectoryolov2::sotfmax(vector<float> &vals) {
	const int _size = vals.size();
	float max_value = *max_element(vals.begin(), vals.end());
	float sum = 0.0;
	for (int i = 0; i < _size; ++i) {
		vals[i] = std::exp(vals[i] - max_value);
		sum += vals[i];
	}
	for (int i = 0; i < _size; ++i) {
		vals[i] = vals[i] / static_cast<float>(sum);
	}
}

void Detectoryolov2::box_to_corner(vector<vector<float>> &boxes) {
	const int _size = boxes.size();
	for (int i = 0; i < _size; ++i) {
		assert(boxes[i].size() == 4);
		float x1 = (boxes[i][0] - boxes[i][2] * 0.5) / float(_featsize);
		float y1 = (boxes[i][1] - boxes[i][3] * 0.5) / float(_featsize);
		float x2 = (boxes[i][0] + boxes[i][2] * 0.5) / float(_featsize);
		float y2 = (boxes[i][1] + boxes[i][3] * 0.5) / float(_featsize);
		boxes[i][0] = x1;
		boxes[i][1] = y1;
		boxes[i][2] = x2;
		boxes[i][3] = y2;
	}
}

void Detectoryolov2::filter_box(vector<vector<float>> &boxes,
                                vector<vector<float>> &fin_boxes,
                                vector<float> &box_conf,
                                vector<float> &scores,
                                vector<vector<float>> &classes_conf,
                                vector<float> &classes_prob) {
	const int _size = boxes.size();
	for (int i = 0; i < _size; ++i) {
		float conf = box_conf[i];
		vector<float> score;
		for (int j = 0; j < _classes_num; ++j) {
			score.push_back(conf * classes_conf[i][j]);
		}
		float max_val = *max_element(score.begin(), score.end());
		int cls_index = max_element(score.begin(), score.end()) - score.begin();
		if (max_val > score_thresh) {
			fin_boxes.push_back(boxes[i]);
			scores.push_back(max_val);
			classes_prob.push_back(cls_index);
		}
	}
}
void Detectoryolov2::draw_bbox_image() {
	cv::Mat image = cv::imread(_image_path);
	const int obj_num = _objectinfo.size();
	for (int i = 0; i < obj_num; ++i)	{
		objectinfo _info = _objectinfo[i];
		cv::Point p1(_info.bbox.x, _info.bbox.y);
		cv::Point p2(_info.bbox.x + _info.bbox.width, _info.bbox.y + _info.bbox.height);
		cv::rectangle(image, p1, p2, cv::Scalar(0, 0, 255), 2, 1, 0);

		char score[20];
		sprintf(score, "%.2f", _info.score);
		string st = score;
		st = _voc_classes[_info.classes] + st;
		cv::Point p3(_info.bbox.x, _info.bbox.y - 5);
		cv::putText(image, st, p3, cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 255, 0));
	}
	cv::imshow("detect", image);
	cv::waitKey();
}