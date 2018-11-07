#include "Detectorsdd.hpp"

Detectorsdd::Detectorsdd(const string &model_path): Detector(model_path),
	_score_thresh(0.5), _nms_thresh(0.5) {
	session = initializer(model_path);
	_out_tensor_names = {
		"ssd_300_vgg/softmax/Reshape_1",
		"ssd_300_vgg/softmax_1/Reshape_1",
		"ssd_300_vgg/softmax_2/Reshape_1",
		"ssd_300_vgg/softmax_3/Reshape_1",
		"ssd_300_vgg/softmax_4/Reshape_1",
		"ssd_300_vgg/softmax_5/Reshape_1",
		"ssd_300_vgg/block4_box/Reshape",
		"ssd_300_vgg/block7_box/Reshape",
		"ssd_300_vgg/block8_box/Reshape",
		"ssd_300_vgg/block9_box/Reshape",
		"ssd_300_vgg/block10_box/Reshape",
		"ssd_300_vgg/block11_box/Reshape",
		"ssd_preprocessing_train/strided_slice"
	};
	_feat_shapes = {38, 19, 10, 5, 3, 1};
	_anchor_sizes = {
		{21, 45},
		{45, 99},
		{99, 153},
		{153, 207},
		{207, 261},
		{261, 315}
	};
	_anchor_ratios = {
		{2, 0.5},
		{2, 0.5, 3, 1.0 / 3},
		{2, 0.5, 3, 1.0 / 3},
		{2, 0.5, 3, 1.0 / 3},
		{2, 0.5},
		{2, 0.5}
	};
	_anchor_steps = {8, 16, 32, 64, 100, 300};
	img_shape = {300, 300};
	prior_scaling = {0.1, 0.1, 0.2, 0.2};
	_anchor_nums = {4, 6, 6, 6, 4, 4};
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


Detectorsdd::~Detectorsdd() {
	session->Close();
}

static void image_to_vector(vector<uint8_t> & data, cv::Mat &image_np) {
	int w = image_np.cols, h = image_np.rows;
	for (int i = 0; i < h; ++i) {
		for (int j = 0; j < w; ++j) {
			for (int k = 0; k < 3; ++k) {
				data.push_back(static_cast<uint8_t>(image_np.at<cv::Vec3b>(i, j)[k]));
			}
		}
	}
}

void Detectorsdd::detect(const string &image_path) {
	_image_path = image_path;
	string input_layer = "Placeholder";

	cv::Mat image = cv::imread(image_path);
	cv::cvtColor(image, image, CV_BGR2RGB);
	int h = image.rows, w = image.cols;
	vector<uint8_t> input_data;
	image_to_vector(input_data, image);

	Tensor input_tensor(tensorflow::DT_UINT8, TensorShape({h, w, 3}));
	auto input_tensor_mapped = input_tensor.tensor<uint8_t, 3>();
	uint8_t* out = input_tensor_mapped.data();
	memcpy(out, &input_data[0], input_data.size()*sizeof(uint8_t));

	vector<Tensor> outputs;
	vector<std::pair<string, Tensor>> inputs = {{input_layer, input_tensor}};
	Status status = session->Run(inputs, _out_tensor_names, {}, &outputs);

	if (!status.ok())
		throw logic_error(status.ToString());
	else
		std::cout << "run model successfully" << std::endl;

	const int shape_num = _feat_shapes.size();
	vector<vector<float>> bboxes;
	vector<float> scores;
	vector<int> classes;

	for (int i = 0; i < shape_num; ++i) {
		const int anchor_num = _anchor_nums[i];
		const int _feat_size = _feat_shapes[i];
		Tensor * predictout = &outputs[i];
		Tensor* locationout = &outputs[i + shape_num];
		auto feat_preditction = predictout->shaped<float, 4>({_feat_size,
		                        _feat_size,
		                        anchor_num, 21
		                                                     });
		auto feat_loactions = locationout->shaped<float, 4>({_feat_size,
		                      _feat_size,
		                      anchor_num, 4
		                                                    });
		vector<vector<float>> anchors = get_anchors(i);


		for (int j = 0; j < _feat_size; ++j) {
			for (int k = 0; k < _feat_size; ++k) {
				for (int m = 0; m < anchor_num; ++m) {
					float xref = (k + 0.5) * _anchor_steps[i] / (img_shape[1]);
					float yref = (j + 0.5) * _anchor_steps[i] / (img_shape[0]);
					float wref = anchors[m][0];
					float href = anchors[m][1];
					float cx = feat_loactions(j, k, m, 0) * wref * prior_scaling[0] + xref;
					float cy = feat_loactions(j, k, m, 1) * href * prior_scaling[1] + yref;
					float w = wref * exp(feat_loactions(j, k, m, 2) * prior_scaling[2]);
					float h = href * exp(feat_loactions(j, k, m, 3) * prior_scaling[3]);
					float _x1 = cx - w / 2.0;
					float _y1 = cy - h / 2.0;
					float _x2 = cx + w / 2.0;
					float _y2 = cy + h / 2.0;
					vector<float> box{_x1, _y1, _x2, _y2};
					vector<float> score;
					for (int c = 1; c < 21; ++c) {
						score.push_back(feat_preditction(j, k, m, c));
					}
					float maxscore = *max_element(score.begin(), score.end());
					int idxes = max_element(score.begin(), score.end()) - score.begin();
					if (maxscore > _score_thresh) {
						bboxes.push_back(box);
						scores.push_back(maxscore);
						classes.push_back(idxes);
					}
				}
			}
		}
	}

	vector<vector<float>> _bboxes;
	vector<float> _scores;
	vector<int> _classes;
	bboxes_sort(bboxes, scores, classes, _bboxes, _scores, _classes);
	vector<cv::Rect> proposals = box_to_corner(_bboxes, w, h);
	vector<int> keep = nms(proposals, _scores, _nms_thresh);
	const int res_num = keep.size();
	for (int i = 0; i < res_num; ++i) {
		objectinfo _info;
		_info.bbox = proposals[keep[i]];
		_info.classes = _classes[keep[i]];
		_info.score = _scores[keep[i]];
		_objectinfo.push_back(_info);
	}
}

vector<vector<float>> Detectorsdd::get_anchors(int index) {
	const int num_anchors = _anchor_nums[index];
	vector<float> sizes = _anchor_sizes[index];
	vector<float> ratios = _anchor_ratios[index];
	vector<vector<float>> anchors(num_anchors);

	for (int i = 0; i < num_anchors; ++i) {
		anchors[i].resize(2);
	}
	int di = 0;
	anchors[di][0] = sizes[0] / img_shape[1];  // anchor_w
	anchors[di][1] = sizes[0] / img_shape[0]; // anchor_h
	di++;
	const int _size_len = sizes.size();
	if (_size_len > 1) {
		anchors[di][0] = sqrt(sizes[0] * sizes[1]) / img_shape[1];
		anchors[di][1] = sqrt(sizes[0] * sizes[1]) / img_shape[0];
		di++;
	}
	const int _ratio_len = ratios.size();
	for (int i = 0; i < _ratio_len; ++i) {
		anchors[di + i][0] = sizes[0] / img_shape[1] * sqrt(ratios[i]);
		anchors[di + i][1] = sizes[0] / img_shape[0] / sqrt(ratios[i]);
	}
	return anchors;
}

void Detectorsdd::bboxes_sort(vector<vector<float>> &bboxes,
                              vector<float> &scores,
                              vector<int> &classes,
                              vector<vector<float>> &_bboxes,
                              vector<float> &_scores,
                              vector<int> &_classes, int topk) {
	vector<int> sortidx = argsort(scores);
	const int nums = sortidx.size() < topk ? sortidx.size() : topk;
	for (int i = 0; i < nums; ++i) {
		_bboxes.push_back(bboxes[sortidx[i]]);
		_scores.push_back(scores[sortidx[i]]);
		_classes.push_back(classes[sortidx[i]]);
	}
}


vector<cv::Rect> Detectorsdd::box_to_corner(vector<vector<float>> bboxes,
        int width,
        int height) {
	vector<cv::Rect> proposals;
	const int box_num = bboxes.size();
	for (int i = 0; i < box_num; ++i) {
		cv::Rect rect;
		rect.x = int(bboxes[i][0] * width);
		rect.y = int(bboxes[i][1] * height);
		rect.width = int(bboxes[i][2] * width - rect.x);
		rect.height = int(bboxes[i][3] * height - rect.y);
		proposals.push_back(rect);
	}
	return proposals;
}


void Detectorsdd::draw_bbox_image() {
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
