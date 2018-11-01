#include "Detectoryolov1.hpp"


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


Config::Config() {
	width = 448;
	height = 448;
	mean_b = 103.939;
	mean_g = 116.779;
	mean_r = 123.68;
}

Detectoryolov1::Detectoryolov1(const string &model_path, float thresh): Detector(model_path),
	_thresh(thresh) {
	session = initializer(model_path);
	std::ifstream in("../yolov1/classes.txt");
	assert(in.is_open());
	int line = 0;
	while (!in.eof()) {
		string _c;
		getline(in, _c);
		if (!_c.empty()) {
			_config.VOC_CLASSES.insert(std::make_pair(line, trim(_c)));
			line++;
		}
	}
}

Detectoryolov1::~Detectoryolov1() {
	session->Close();
}

cv::Mat Detectoryolov1::preprocess(const string &image_path) {
	cv::Mat image = cv::imread(image_path);
	int h = image.rows, w = image.cols;
	image.convertTo(image, CV_32FC3);
	for (int i = 0; i < h; ++i) {
		for (int j = 0; j < w; ++j) {
			image.at<cv::Vec3f>(i, j)[0] -= _config.mean_b;
			image.at<cv::Vec3f>(i, j)[1] -= _config.mean_g;
			image.at<cv::Vec3f>(i, j)[2] -= _config.mean_r;
		}
	}
	cv::Mat dstimage;
	cv::resize(image, dstimage, cv::Size(_config.width, _config.height));
	return dstimage;
}


void Detectoryolov1::detect(const string &image_path) {
	_image_path = image_path;
	string image_layer = "input";
	string bool_layer = "is_training";
	string output_layer = "output";

	cv::Mat image = preprocess(image_path);

	vector<float> input_image;
	image_to_vector(input_image, image);
	Tensor input_tensor(tensorflow::DT_FLOAT, TensorShape({1, _config.height, _config.width, 3}));
	auto input_tensor_mapped = input_tensor.tensor<float, 4>();
	float* out = input_tensor_mapped.data();
	memcpy(out, &input_image[0], input_image.size()*sizeof(float));

	Tensor bool_tensor(tensorflow::DT_BOOL, TensorShape());
	bool_tensor.scalar<bool>()() = false;

	vector<Tensor> outputs;
	vector<std::pair<string, Tensor>> inputs = {{image_layer, input_tensor},
		{bool_layer, bool_tensor}
	};
	vector<string> out_tensor_names = {output_layer};
	Status status = session->Run(inputs, out_tensor_names, {}, &outputs);
	if (!status.ok())
		throw logic_error(status.ToString());
	else
		std::cout << "run model successfully" << std::endl;

	Tensor* detout = &outputs[0];
	auto predicts = detout->shaped<float, 3>({14, 14, 30});

	vector<vector<float>> obj_prob;
	vector<vector<float>> boxes;
	vector<vector<float>> cls_prob;

	const int grid_num = 14;
	float cell_size = 1 / static_cast<float>(grid_num);

	for (int i = 0; i < 14; ++i) {
		for (int j = 0; j < 14; ++j) {
			vector<float> cur_cls;
			vector<float> cur_prob{predicts(i, j, 4), predicts(i, j, 9)};
			for (int k = 10; k < 30; ++k)
				cur_cls.push_back(predicts(i, j, k));
			cls_prob.push_back(cur_cls);
			obj_prob.push_back(cur_prob);
		}
	}

	vector<vector<float>> det_boxes;
	vector<int> classes;
	vector<float> scores;

	for (int i = 0; i < grid_num; ++i) {
		for (int j = 0; j < grid_num; ++j) {
			int index = i * grid_num + j;
			for (int k = 0; k < 2; ++k) {
				if (obj_prob[index][k] > _thresh) {
					float x1 = predicts(i, j, 5 * k) * cell_size + j * cell_size;
					float y1 = predicts(i, j, 5 * k + 1) * cell_size + i * cell_size;
					float x2 = x1 + 0.5 * predicts(i, j, 5 * k + 2);
					float y2 = y1 + 0.5 * predicts(i, j, 5 * k + 3);
					x1 -= 0.5 * predicts(i, j, 5 * k + 2);
					y1 -= 0.5 * predicts(i, j, 5 * k + 3);

					vector<float> box{x1, y1, x2, y2};
					int cls_index = max_element(cls_prob[index].begin(), cls_prob[index].end()) -
					                cls_prob[index].begin();
					float max_prob = *max_element(cls_prob[index].begin(), cls_prob[index].end());
					float con_prob = obj_prob[index][k];
					if ((max_prob * con_prob) > _thresh) {
						det_boxes.push_back(box);
						classes.push_back(cls_index);
						scores.push_back(con_prob * max_prob);
					}
				}
			}
		}
	}

	const long boxes_num = det_boxes.size();
	cv::Mat image_ = cv::imread(image_path);
	int h = image_.rows, w = image_.cols;
	vector<cv::Rect> proposals(boxes_num);
	for (int i = 0; i < boxes_num; ++i) {
		proposals[i].x = static_cast<int>(det_boxes[i][0] * w);
		proposals[i].y = static_cast<int>(det_boxes[i][1] * h);
		proposals[i].width = static_cast<int>(det_boxes[i][2] * w) - proposals[i].x;
		proposals[i].height = static_cast<int>(det_boxes[i][3] * h) - proposals[i].y;
	}
	vector<int> keep = nms(proposals, scores, 0.5);
	const long res_num = keep.size();
	for (int i = 0; i < res_num; ++i) {
		OjbectInfo _info;
		_info.bbox = proposals[keep[i]];
		_info.classes = classes[keep[i]];
		_info.score = scores[keep[i]];
		_objectinfo.push_back(_info);
	}
}



void Detectoryolov1::draw_bbox_image() {
	cv::Mat image = cv::imread(_image_path);
	const int obj_num = _objectinfo.size();
	for (int i = 0; i < obj_num; ++i)	{
		OjbectInfo _info = _objectinfo[i];
		cv::Point p1(_info.bbox.x, _info.bbox.y);
		cv::Point p2(_info.bbox.x + _info.bbox.width, _info.bbox.y + _info.bbox.height);
		cv::rectangle(image, p1, p2, cv::Scalar(0, 0, 255), 2, 1, 0);

		char score[20];
		sprintf(score, "%.2f", _info.score);
		string st = score;
		st = _config.VOC_CLASSES[_info.classes] + st;
		cv::Point p3(_info.bbox.x, _info.bbox.y - 5);
		cv::putText(image, st, p3, cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 255, 0));
	}

	cv::imshow("detect", image);
	cv::waitKey();
}

vector<OjbectInfo> Detectoryolov1::get_objectInfo() {
	return _objectinfo;
}