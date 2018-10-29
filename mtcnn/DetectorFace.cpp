#include "DetectorFace.hpp"


static void image_to_vector(vector<float>& data, const cv::Mat &image_np) {
	int w = image_np.cols, h = image_np.rows;
	for (int i = 0; i < h; ++i) {
		for (int j = 0; j < w; ++j) {
			for (int k = 0; k < 3; ++k) {
				data.push_back(static_cast<float>(image_np.at<cv::Vec3b>(i, j)[k]));
			}
		}
	}
}

facedetector::facedetector(const string &model_path): Detector(model_path),
	_min_size(24), _factor(0.709),
	_th1(0.6), _th2(0.7), _th3(0.7) {
	session =  initializer(model_path);
}

facedetector::~facedetector() {
	session->Close();
}

void facedetector::set_min_size(int min_size) {
	_min_size = min_size;
}
void facedetector::set_factor(float factor) {
	_factor = factor;
}
void facedetector::set_threshold(float th1, float th2, float th3) {
	_th1 = th1;
	_th2 = th2;
	_th3 = th3;
}


void facedetector::detect(const string &image_path) {
	_image_path = image_path;
	string image_layer = "input";
	string min_size_layer = "min_size";
	string thresh_layer = "thresholds";
	string factor_layer = "factor";
	string out_prob_layer = "prob";
	string out_landmarks_layer = "landmarks";
	string out_box_layer = "box";

	cv::Mat image = cv::imread(image_path);
	int w = image.cols, h = image.rows;
	vector<float> input_image;
	image_to_vector(input_image, image);

	Tensor input_image_tensor(tensorflow::DT_FLOAT, TensorShape({h, w, 3}));
	auto image_tensor_mapped = input_image_tensor.tensor<float, 3>();
	float* out = image_tensor_mapped.data();
	memcpy(out, &input_image[0], input_image.size()*sizeof(float));

	Tensor min_size_tensor(tensorflow::DT_FLOAT, TensorShape({}));
	min_size_tensor.scalar<float>()() = _min_size;

	Tensor factor_tensor(tensorflow::DT_FLOAT, TensorShape({}));
	factor_tensor.scalar<float>()() = _factor;

	vector<float> input_thresh{_th1, _th2, _th3};
	Tensor thresh_tensor(tensorflow::DT_FLOAT, TensorShape({3}));
	auto thresh_tensor_mapped = thresh_tensor.tensor<float, 1>();
	float* thresh_out = thresh_tensor_mapped.data();
	memcpy(thresh_out, &input_thresh[0], input_thresh.size()*sizeof(float));

	vector<Tensor> outputs;
	vector<std::pair<string, Tensor>> inputs = {{image_layer, input_image_tensor},
		{min_size_layer, min_size_tensor}, {factor_layer, factor_tensor},
		{thresh_layer, thresh_tensor}
	};
	vector<string> output_tensor_names = {out_prob_layer, out_landmarks_layer,
	                                      out_box_layer
	                                     };
	Status status = session->Run(inputs, output_tensor_names, {}, &outputs);
	if (!status.ok())
		throw logic_error(status.ToString());
	else
		std::cout << "run model successfully" << std::endl;

	Tensor* output_prob = &outputs[0];
	Tensor* output_landmarks = &outputs[1];
	Tensor* output_boxes = &outputs[2];


	const Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>,
	      Eigen::Aligned>& predict_prob = output_prob->flat<float>();
	const Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>,
	      Eigen::Aligned>& predict_landmarks = output_landmarks->flat<float>();
	const Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>,
	      Eigen::Aligned>& predict_boxes = output_boxes->flat<float>();

	const long boxes_num = predict_boxes.size();
	const long landmarks_num = predict_landmarks.size();
	const long prob_num = predict_prob.size();
	std::cout << boxes_num << " " << landmarks_num << " " << prob_num << std::endl;

	vector<float> res_prob(prob_num, 0);
	vector<float> res_boxes(boxes_num, 0);
	vector<float> res_landmarks(landmarks_num, 0);
	for (int i = 0; i < prob_num; i++) res_prob[i] = predict_prob(i);
	for (int i = 0; i < boxes_num; i++) res_boxes[i] = predict_boxes(i);
	for (int i = 0; i < landmarks_num; i++) res_landmarks[i] = predict_landmarks(i);

	for (int i = 0; i < prob_num; ++i) {
		FaceInfo faceinfo;
		faceinfo.score = res_prob[i];
		for (int j = 0; j < 4; ++j) {
			faceinfo.bbox[j] = res_boxes[i * 4 + j];
		}
		for (int j = 0; j < 10; ++j) {
			faceinfo.landmark[j] = res_landmarks[i * 10 + j];
		}
		det_faceinfo.push_back(faceinfo);
	}
}


void facedetector::draw_bbox_image() {
	cv::Mat image = cv::imread(_image_path);
	long long face_num = det_faceinfo.size();
	for (int i = 0; i < face_num; ++i) {
		cv::Point p1(det_faceinfo[i].bbox[1], det_faceinfo[i].bbox[0]);
		cv::Point p2(det_faceinfo[i].bbox[3], det_faceinfo[i].bbox[2]);
		cv::rectangle(image, p1, p2, cv::Scalar(0, 0, 255), 2, 1, 0);

		for (int j = 0; j < 5; ++j) {
			cv::Point p3(det_faceinfo[i].landmark[j + 5], det_faceinfo[i].landmark[j]);
			cv::circle(image, p3, 1, cv::Scalar(0, 255, 0), 2);
		}
		string score_text = std::to_string(det_faceinfo[i].score);
		cv::Point p4(det_faceinfo[i].bbox[1], det_faceinfo[i].bbox[0] - 5);
		cv::putText(image, score_text, p4, cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 0, 255));
	}
	cv::imshow("face", image);
	cv::waitKey();
}


vector<FaceInfo> facedetector::get_FaceInfo() {
	return det_faceinfo;
}