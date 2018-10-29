#ifndef DETECTORHAND_H_
#define DETECTORHAND_H_

#include "Detector.hpp"


struct handInfo {
	float score;
	float bbox[4];
};


class DetectorHand: public Detector
{
private:
	Session *session;
	double _thresh;
	vector<handInfo> det_hand_Info;
	string _image_path;

public:
	DetectorHand(const string &model_path, double thresh = 0.2);
	~DetectorHand();

	virtual void detect(const string &image_path);

	virtual void draw_bbox_image();

	vector<handInfo> get_handInfo();


};
#endif
