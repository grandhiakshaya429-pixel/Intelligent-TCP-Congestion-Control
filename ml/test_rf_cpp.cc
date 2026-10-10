#include <onnxruntime_cxx_api.h>
#include <array>
#include <iostream>
#include <vector>

int main()
{
    try
    {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "RF_Test");
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(1);

        Ort::Session session(env, "congestion_model.onnx", options);

        std::array<int64_t, 2> inputShape = {1, 8};
        std::array<float, 8> features = {
            10.0f, 25.0f, 10000.0f, 500000.0f,
            8.0f, 60.0f, 0.02f, 8.0f
        };

        auto memoryInfo =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        auto inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo,
            features.data(),
            features.size(),
            inputShape.data(),
            inputShape.size()
        );

        const char* inputNames[] = {"float_input"};
        const char* outputNames[] = {"output_label"};

        auto outputs = session.Run(
            Ort::RunOptions{nullptr},
            inputNames,
            &inputTensor,
            1,
            outputNames,
            1
        );

        auto labels = outputs[0].GetTensorData<std::string>();
        std::cout << "Random Forest C++ inference: SUCCESS\n";
        std::cout << "Predicted congestion: " << labels[0] << '\n';

        std::string label = labels[0];
        if (label != "Low" && label != "Medium" && label != "High")
        {
            std::cerr << "FAIL: Unexpected congestion label.\n";
            return 1;
        }

        std::cout << "PASS: Valid congestion label returned.\n";
        return 0;
    }
    catch (const Ort::Exception& e)
    {
        std::cerr << "ONNX Runtime error: " << e.what() << '\n';
        return 1;
    }
}
