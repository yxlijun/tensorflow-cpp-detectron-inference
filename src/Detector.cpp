#include "Detector.hpp"


Detector::Detector(const string & model_path) {

}

Detector::~Detector() {
}


Session* Detector::initializer(const string & model_path) {
	Session *session;
	Status status = tensorflow::NewSession(SessionOptions(), &session);
	if (!status.ok())
		throw logic_error(status.ToString());
	else
		std::cout << "Session created successfully" << std::endl;
	GraphDef graph_def;
	status = ReadBinaryProto(Env::Default(), model_path, &graph_def);
	if (!status.ok())
		throw logic_error(status.ToString());
	else
		std::cout << "Load graph protobuf successfully" << std::endl;

	status = session->Create(graph_def);
	if (!status.ok())
		throw logic_error(status.ToString());
	else
		std::cout << "Add graph to session successfully" << std::endl;
	return session;
}



