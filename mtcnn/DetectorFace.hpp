#ifndef DETECTORFACE_HPP_
#define DETECTORFACE_HPP_

#include "Detector.hpp"
#include "utils.hpp"

struct FaceInfo {
	// y1,x1,y2,x2
	float bbox[4];
	float landmark[10];
	float score;
};


class facedetector: public Detector
{
public:
	facedetector(const string &model_path);

	~facedetector();

	virtual void detect(const string &image_path);

	virtual void draw_bbox_image();

	void set_min_size(int min_size);

	void set_factor(float factor);

	void set_threshold(float th1, float th2, float th3);

	vector<FaceInfo> get_FaceInfo();
private:
	int _min_size;
	float _factor;
	float _th1, _th2, _th3;
	vector<FaceInfo> det_faceinfo;
	Session* session;
	string _image_path;

};
#endif


