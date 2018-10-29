#include "DetectorHand.hpp"


static void image_to_vector(vector<uint8_t>& data, const cv::Mat &image_np) {
	int w = image_np.cols, h = image_np.rows;
	for (int i = 0; i < h; ++i) {
		for (int j = 0; j < w; ++j) {
			for (int k = 0; k < 3; ++k) {
				data.push_back(static_cast<uint8_t>(image_np.at<cv::Vec3b>(i, j)[k]));
			}
		}
	}
}


DetectorHand::DetectorHand(const string &model_path, double thresh):
	Detector(model_path), _thresh(thresh) {
	session =  initializer(model_path);
}

DetectorHand::~DetectorHand() {
	session->Close();
}


void DetectorHand::detect(const string &image_path) {
	_image_path = image_path;
	string input_layer = "image_tensor";
	string output_boxes_layer  = "detection_boxes";
	string output_score_layer = "detection_scores";

	cv::Mat image = cv::imread(image_path);
	cv::Mat _image;
	cv::cvtColor(image, _image, CV_BGR2RGB);
	vector<uint8_t> input_data;
	int w = _image.cols, h = _image.rows;
	image_to_vector(input_data, _image);

	Tensor input_tensor(tensorflow::DT_UINT8, TensorShape({1, h, w, 3}));
	auto input_tensor_mapped = input_tensor.tensor<uint8_t, 4>();
	uint8_t* out = input_tensor_mapped.data();
	memcpy(out, &input_data[0], input_data.size()*sizeof(uint8_t));

	vector<Tensor> outputs;
	vector<std::pair<string, Tensor>> inputs = {{input_layer, input_tensor}};
	vector<string> output_tensor_names = {output_score_layer, output_boxes_layer};

	Status status = session->Run(inputs, output_tensor_names, {}, &outputs);

	if (!status.ok())
		throw logic_error(status.ToString());
	else
		std::cout << "run model successfully" << std::endl;

	Tensor* output_score = &outputs[0];
	Tensor* output_boxes = &outputs[1];

	const Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>,
	      Eigen::Aligned>& predict_score = output_score->flat<float>();
	const Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>,
	      Eigen::Aligned>& predict_boxes = output_boxes->flat<float>();

	const long boxes_num = predict_boxes.size();
	const long socre_num = predict_score.size();

	vector<float> score_res(socre_num, 0);
	vector<float> boxes_res(boxes_num, 0);
	for (int i = 0; i < socre_num; ++i) score_res[i] = predict_score(i);
	for (int i = 0; i < boxes_num; ++i) boxes_res[i] = predict_boxes(i);
	for (int i = 0; i < socre_num; ++i) {
		if (score_res[i] > _thresh) {
			handInfo curinfo;
			curinfo.score = score_res[i];
			int top = static_cast<int>(boxes_res[i * 4 + 0] * h);
			int left = static_cast<int>(boxes_res[i * 4 + 1] * w);
			int bottom = static_cast<int>(boxes_res[i * 4 + 2] * h);
			int right = static_cast<int>(boxes_res[i * 4 + 3] * w);
			curinfo.bbox[0] = left;
			curinfo.bbox[1] = top;
			curinfo.bbox[2] = right;
			curinfo.bbox[3] = bottom;
			det_hand_Info.push_back(curinfo);
		}
	}
}

void DetectorHand::draw_bbox_image() {
	cv::Mat image = cv::imread(_image_path);
	int hand_num = det_hand_Info.size();
	for (int i = 0; i < hand_num; ++i) {
		cv::Point p1(det_hand_Info[i].bbox[0], det_hand_Info[i].bbox[1]);
		cv::Point p2(det_hand_Info[i].bbox[2], det_hand_Info[i].bbox[3]);
		cv::rectangle(image, p1, p2, cv::Scalar(0, 255, 0), 1, 1, 0);
	}
	cv::imshow("detect", image);
	cv::waitKey();
}

vector<handInfo> DetectorHand::get_handInfo() {
	return det_hand_Info;
}
