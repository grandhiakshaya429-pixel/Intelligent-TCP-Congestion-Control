#include <onnxruntime_cxx_api.h>
#include <array>
#include <iostream>
#include <vector>

int main()
{
    try
    {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "PPO_Test");
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(1);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        Ort::Session session(
            env,
            "ppo_policy.onnx",
            options
        );

        std::array<int64_t, 2> inputShape = {1, 7};
        std::array<float, 7> observation = {
            0.5f, 0.5f, 0.0f, 0.5f, 0.3f, 0.0f, 0.5f
        };

        Ort::MemoryInfo memoryInfo =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo,
            observation.data(),
            observation.size(),
            inputShape.data(),
            inputShape.size()
        );

        const char* inputNames[] = {"observation"};
        const char* outputNames[] = {"action"};

        auto outputs = session.Run(
            Ort::RunOptions{nullptr},
            inputNames,
            &inputTensor,
            1,
            outputNames,
            1
        );

        int64_t action = outputs[0].GetTensorData<int64_t>()[0];

        std::cout << "C++ ONNX inference: SUCCESS\n";
        std::cout << "Predicted action: " << action << '\n';

        if (action < 0 || action > 2)
        {
            std::cerr << "FAIL: Unexpected action value.\n";
            return 1;
        }

        std::cout << "PASS: Action is valid (0=Decrease, 1=Maintain, 2=Increase).\n";
        return 0;
    }
    catch (const Ort::Exception& e)
    {
        std::cerr << "ONNX Runtime error: " << e.what() << '\n';
        return 1;
    }
}
