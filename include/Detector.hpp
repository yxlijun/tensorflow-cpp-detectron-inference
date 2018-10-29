#ifndef DETECTOR_HPP_
#define DETECTOR_HPP_

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/cc/ops/const_op.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <exception>
#include <initializer_list>

using std::vector;
using std::string;
using std::logic_error;
using std::runtime_error;
using std::initializer_list;
using tensorflow::Env;
using tensorflow::Session;
using tensorflow::Status;
using tensorflow::GraphDef;
using tensorflow::NewSession;
using tensorflow::SessionOptions;
using tensorflow::Tensor;
using tensorflow::TensorShape;


class Detector
{
public:
	Detector(const string & model_path);

	virtual ~Detector();

	Session* initializer(const string & model_path);

	virtual void detect(const string &image_path) = 0;

	virtual void draw_bbox_image() = 0;
};
#endif