#include "utils.hpp"


float iou(const cv::Rect& r1, const cv::Rect &r2) {
	int x1 = std::max(r1.x, r2.x);
	int y1 = std::max(r1.y, r2.y);
	int x2 = std::min(r1.x + r1.width, r2.x + r2.width);
	int y2 = std::min(r1.y + r1.height, r2.y + r2.height);
	int w = std::max(0, x2 - x1);
	int h = std::max(0, y2 - y1);
	float inter = w * h;
	float ratio = inter / (r1.area() + r2.area() - inter);
	return (ratio >= 0) ? ratio : 0;
}

static vector<int> argsort(vector<float> scores) {
	vector<float> _scores = scores;
	std::sort(scores.begin(), scores.end(), std::greater<float>());
	int _size = scores.size();
	vector<int> ids;
	for (int i = 0; i < _size; ++i) {
		auto pos = std::find(_scores.begin(), _scores.end(), scores[i]);
		ids.push_back(pos - _scores.begin());
	}
	return ids;
}

vector<int> nms(vector<cv::Rect> & proposal, vector<float> scores, const float thresh) {
	vector<int> order = argsort(scores);
	vector<int> keep;
	while (!order.empty()) {
		int last = order[0];
		keep.push_back(last);
		long _size = order.size();
		if (_size == 1) break;
		vector<int> keep_order;
		for (int j = 1; j < _size ; ++j) {
			float ratio = iou(proposal[last], proposal[order[j]]);
			if (ratio <= thresh)
				keep_order.push_back(order[j]);
		}
		order = keep_order;
	}
	return keep;
}

string& trim(string &str) {
	if (str.empty()) {
		return str;
	}
	str.erase(0, str.find_first_not_of(" "));
	str.erase(str.find_last_not_of(" ") + 1);
	return str;
}