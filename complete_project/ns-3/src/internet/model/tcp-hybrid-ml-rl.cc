#include "tcp-hybrid-ml-rl.h"

#include "ns3/double.h"
#include "ns3/log.h"
#include "ns3/string.h"
#include "ns3/simulator.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <memory>
#include <string>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("TcpHybridMlRl");
NS_OBJECT_ENSURE_REGISTERED(TcpHybridMlRl);

namespace
{
Ort::Env&
GetOnnxEnvironment()
{
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Ns3HybridController");
    return env;
}
} // namespace

struct TcpHybridMlRl::ModelBundle
{
    Ort::Session rfSession;
    Ort::Session ppoSession;

    explicit ModelBundle(const std::string& directory)
        : rfSession(
              GetOnnxEnvironment(),
              (directory + "/congestion_model.onnx").c_str(),
              CreateOptions()),
          ppoSession(
              GetOnnxEnvironment(),
              (directory + "/ppo_policy.onnx").c_str(),
              CreateOptions())
    {
        NS_LOG_INFO("Loaded Random Forest and PPO ONNX models from "
                    << directory);
    }

    static Ort::SessionOptions CreateOptions()
    {
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(1);
        options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);
        return options;
    }
};

TypeId
TcpHybridMlRl::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::TcpHybridMlRl")
            .SetParent<TcpCongestionOps>()
            .AddConstructor<TcpHybridMlRl>()
            .AddAttribute(
                "ModelDirectory",
                "Directory containing congestion_model.onnx and ppo_policy.onnx.",
                StringValue("../ml"),
                MakeStringAccessor(&TcpHybridMlRl::m_modelDirectory),
                MakeStringChecker())
            .AddAttribute(
                "DecisionLogPath",
                "CSV path for Random Forest and PPO decisions.",
                StringValue("../results/csv/hybrid_decisions.csv"),
                MakeStringAccessor(&TcpHybridMlRl::m_decisionLogPath),
                MakeStringChecker())
            .AddAttribute(
                "LinkBandwidthMbps",
                "Configured link bandwidth in Mbps.",
                DoubleValue(10.0),
                MakeDoubleAccessor(&TcpHybridMlRl::m_linkBandwidthMbps),
                MakeDoubleChecker<double>(0.001))
            .AddAttribute(
                "LinkDelayMs",
                "Configured point-to-point link delay in milliseconds.",
                DoubleValue(50.0),
                MakeDoubleAccessor(&TcpHybridMlRl::m_linkDelayMs),
                MakeDoubleChecker<double>(0.0))
            .SetGroupName("Internet");

    return tid;
}

TcpHybridMlRl::TcpHybridMlRl()
    : TcpCongestionOps(),
      m_latestRtt(Seconds(0)),
      m_minRtt(Time::Max()),
      m_lastCwnd(0),
      m_lastLossEvent(0),
      m_mlCongestionState(0),
      m_models(),
      m_modelDirectory("../ml"),
      m_decisionLogPath("../results/csv/hybrid_decisions.csv"),
      m_linkBandwidthMbps(10.0),
      m_linkDelayMs(50.0),
      m_lastDecisionTime(-1.0),
      m_lastThroughputMbps(0.0),
      m_totalAckedBytes(0),
      m_lastDecisionAckedBytes(0),
      m_cwndFactor(1.0),
      m_lastLossCount(0),
      m_lastPpoAction(1)
{
    m_models = std::make_shared<ModelBundle>(m_modelDirectory);
}

TcpHybridMlRl::TcpHybridMlRl(const TcpHybridMlRl& other)
    : TcpCongestionOps(other),
      m_latestRtt(other.m_latestRtt),
      m_minRtt(other.m_minRtt),
      m_lastCwnd(other.m_lastCwnd),
      m_lastLossEvent(other.m_lastLossEvent),
      m_mlCongestionState(other.m_mlCongestionState),
      m_models(other.m_models),
      m_modelDirectory(other.m_modelDirectory),
      m_decisionLogPath(other.m_decisionLogPath),
      m_linkBandwidthMbps(other.m_linkBandwidthMbps),
      m_linkDelayMs(other.m_linkDelayMs),
      m_lastDecisionTime(other.m_lastDecisionTime),
      m_lastThroughputMbps(other.m_lastThroughputMbps),
      m_totalAckedBytes(other.m_totalAckedBytes),
      m_lastDecisionAckedBytes(other.m_lastDecisionAckedBytes),
      m_cwndFactor(other.m_cwndFactor),
      m_lastLossCount(other.m_lastLossCount),
      m_lastPpoAction(other.m_lastPpoAction)
{
}

TcpHybridMlRl::~TcpHybridMlRl() = default;

std::string
TcpHybridMlRl::GetName() const
{
    return "TcpHybridMlRl";
}

void
TcpHybridMlRl::PktsAcked(Ptr<TcpSocketState> tcb,
                         uint32_t segmentsAcked,
                         const Time& rtt)
{
    NS_LOG_FUNCTION(this << tcb << segmentsAcked << rtt);

    if (segmentsAcked > 0)
    {
        m_totalAckedBytes +=
            static_cast<uint64_t>(segmentsAcked) * tcb->m_segmentSize;
    }

    if (!rtt.IsZero())
    {
        m_latestRtt = rtt;
        m_minRtt = std::min(m_minRtt, rtt);
    }
}

void
TcpHybridMlRl::CwndEvent(Ptr<TcpSocketState> tcb,
                         const TcpSocketState::TcpCAEvent_t event)
{
    NS_LOG_FUNCTION(this << tcb << event);

    if (event == TcpSocketState::CA_EVENT_LOSS)
    {
        m_lastLossEvent = 1;
        ++m_lastLossCount;
    }
}

void
TcpHybridMlRl::CongestionStateSet(
    Ptr<TcpSocketState> tcb,
    const TcpSocketState::TcpCongState_t newState)
{
    NS_LOG_FUNCTION(this << tcb << newState);

    if (newState == TcpSocketState::CA_RECOVERY ||
        newState == TcpSocketState::CA_LOSS)
    {
        m_lastLossEvent = 1;
    }
}

void
TcpHybridMlRl::IncreaseWindow(Ptr<TcpSocketState> tcb,
                              uint32_t segmentsAcked)
{
    NS_LOG_FUNCTION(this << tcb << segmentsAcked);

    if (tcb->m_cWnd.Get() == 0u || segmentsAcked == 0)
    {
        return;
    }

    const double now = Simulator::Now().GetSeconds();

    // Between model decisions, use bounded NewReno-style growth.
    // When RF detects High congestion, suppress growth.
    if (m_lastDecisionTime >= 0.0 &&
        now - m_lastDecisionTime < 0.100)
    {
        if (m_mlCongestionState == 2)
        {
            return;
        }

        const uint32_t mss = tcb->m_segmentSize;
        uint32_t remainingAcked = segmentsAcked;

        // NewReno-style slow start: consume at most one segment
        // of acknowledged data for the slow-start increase.
        if (tcb->m_cWnd < tcb->m_ssThresh && remainingAcked > 0)
        {
            tcb->m_cWnd += mss;
            --remainingAcked;
        }

        // If acknowledged segments remain and the threshold has
        // been reached, apply congestion avoidance.
        if (tcb->m_cWnd >= tcb->m_ssThresh && remainingAcked > 0)
        {
            const double adder =
                static_cast<double>(mss) * mss / tcb->m_cWnd.Get();

            tcb->m_cWnd += static_cast<uint32_t>(
                std::max(1.0, adder));
        }
        return;
    }

    const double elapsed =
        (m_lastDecisionTime < 0.0)
            ? 0.100
            : std::max(0.001, now - m_lastDecisionTime);

    const uint64_t newlyAckedBytes =
        (m_totalAckedBytes >= m_lastDecisionAckedBytes)
            ? m_totalAckedBytes - m_lastDecisionAckedBytes
            : 0;

    m_lastThroughputMbps =
        static_cast<double>(newlyAckedBytes) * 8.0 /
        elapsed / 1000000.0;

    const double acknowledgedPackets =
        static_cast<double>(m_totalAckedBytes) /
        std::max(1u, tcb->m_segmentSize);

    const double lossRatePercent =
        100.0 * static_cast<double>(m_lastLossCount) /
        std::max(1.0, acknowledgedPackets +
                         static_cast<double>(m_lastLossCount));

    const double rttMs = m_latestRtt.GetMilliSeconds();
    const double cwndBytes = tcb->m_cWnd.Get();

    /*
     * RF feature order matches train_random_forest.py:
     * bandwidth, configured delay, CWND, acknowledged bytes,
     * interval throughput, RTT proxy, loss-rate proxy, loss-event count.
     */
    std::array<float, 8> rfFeatures = {
        static_cast<float>(m_linkBandwidthMbps),
        static_cast<float>(m_linkDelayMs),
        static_cast<float>(cwndBytes),
        static_cast<float>(m_totalAckedBytes),
        static_cast<float>(m_lastThroughputMbps),
        static_cast<float>(rttMs),
        static_cast<float>(lossRatePercent),
        static_cast<float>(m_lastLossCount)
    };

    const std::array<int64_t, 2> rfShape = {1, 8};

    Ort::MemoryInfo memoryInfo =
        Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator,
            OrtMemTypeDefault);

    Ort::Value rfTensor = Ort::Value::CreateTensor<float>(
        memoryInfo,
        rfFeatures.data(),
        rfFeatures.size(),
        rfShape.data(),
        rfShape.size());

    const char* rfInputNames[] = {"float_input"};
    const char* rfOutputNames[] = {"output_label"};

    auto rfOutputs = m_models->rfSession.Run(
        Ort::RunOptions{nullptr},
        rfInputNames,
        &rfTensor,
        1,
        rfOutputNames,
        1);

    const std::string label =
        rfOutputs[0].GetTensorData<std::string>()[0];

    if (label == "Low")
    {
        m_mlCongestionState = 0;
    }
    else if (label == "Medium")
    {
        m_mlCongestionState = 1;
    }
    else if (label == "High")
    {
        m_mlCongestionState = 2;
    }
    else
    {
        NS_FATAL_ERROR("Unexpected Random Forest label: " << label);
    }

    /*
     * PPO observation order matches ppo_environment.py:
     * normalized bandwidth, delay, CWND factor, throughput,
     * RTT proxy, loss-rate proxy, and RF congestion state.
     */
    std::array<float, 7> observation = {
        static_cast<float>(
            std::clamp(m_linkBandwidthMbps / 10.0, 0.0, 1.0)),
        static_cast<float>(
            std::clamp(m_linkDelayMs / 150.0, 0.0, 1.0)),
        static_cast<float>(
            std::clamp(m_cwndFactor - 0.5, 0.0, 1.0)),
        static_cast<float>(
            std::clamp(m_lastThroughputMbps / 10.0, 0.0, 1.0)),
        static_cast<float>(
            std::clamp(rttMs / 500.0, 0.0, 1.0)),
        static_cast<float>(
            std::clamp(lossRatePercent / 100.0, 0.0, 1.0)),
        static_cast<float>(m_mlCongestionState / 2.0)
    };

    const std::array<int64_t, 2> ppoShape = {1, 7};

    Ort::Value ppoTensor = Ort::Value::CreateTensor<float>(
        memoryInfo,
        observation.data(),
        observation.size(),
        ppoShape.data(),
        ppoShape.size());

    const char* ppoInputNames[] = {"observation"};
    const char* ppoOutputNames[] = {"action"};

    auto ppoOutputs = m_models->ppoSession.Run(
        Ort::RunOptions{nullptr},
        ppoInputNames,
        &ppoTensor,
        1,
        ppoOutputNames,
        1);

    int64_t action =
        ppoOutputs[0].GetTensorData<int64_t>()[0];

    if (action < 0 || action > 2)
    {
        NS_FATAL_ERROR("Unexpected PPO action: " << action);
    }

    // Hybrid safety rule: do not increase CWND while RF predicts High.
    if (m_mlCongestionState == 2 && action == 2)
    {
        action = 1;
    }

    m_lastPpoAction = static_cast<uint32_t>(action);

    const double oldFactor = m_cwndFactor;

    if (action == 0)
    {
        m_cwndFactor = std::max(0.5, m_cwndFactor * 0.8);
    }
    else if (action == 2)
    {
        m_cwndFactor = std::min(1.5, m_cwndFactor * 1.2);
    }

    // Apply the change in the learned factor, rather than repeatedly
    // multiplying CWND by 1.2 on every ACK.
    if (m_cwndFactor != oldFactor)
    {
        const double scaledCwnd =
            static_cast<double>(tcb->m_cWnd.Get()) *
            m_cwndFactor / oldFactor;

        const uint32_t newCwnd = std::max(
            2 * tcb->m_segmentSize,
            static_cast<uint32_t>(scaledCwnd));

        tcb->m_cWnd = newCwnd;
    }

    m_lastCwnd = tcb->m_cWnd.Get();
    m_lastDecisionTime = now;
    m_lastDecisionAckedBytes = m_totalAckedBytes;
    m_lastLossEvent = 0;

    const char* actionName =
        (m_lastPpoAction == 0) ? "Decrease" :
        (m_lastPpoAction == 1) ? "Maintain" : "Increase";

    std::ofstream decisionFile(
        m_decisionLogPath,
        std::ios::out | std::ios::app);

    if (!decisionFile.is_open())
    {
        NS_LOG_WARN("Could not open decision log: " << m_decisionLogPath);
    }
    else
    {
        if (decisionFile.tellp() == std::streampos(0))
        {
            decisionFile
                << "TimeSeconds,RF_Classification,PPO_Action,"
                << "CWND_Bytes,RTT_Ms,Throughput_Mbps,"
                << "LossRatePercent,AckedBytes\n";
        }

        decisionFile << std::fixed << std::setprecision(6)
                     << now << "," << label << "," << actionName << ","
                     << tcb->m_cWnd.Get() << "," << rttMs << ","
                     << m_lastThroughputMbps << "," << lossRatePercent << ","
                     << m_totalAckedBytes << "\n";
    }

    NS_LOG_INFO(
        "Hybrid decision: RF=" << label
        << ", PPO action=" << actionName
        << ", CWND=" << tcb->m_cWnd.Get());
}

uint32_t
TcpHybridMlRl::GetSsThresh(Ptr<const TcpSocketState> tcb,
                           uint32_t bytesInFlight)
{
    NS_LOG_FUNCTION(this << tcb << bytesInFlight);

    return std::max(
        2 * tcb->m_segmentSize,
        static_cast<uint32_t>(tcb->m_cWnd.Get() * 0.5));
}

Ptr<TcpCongestionOps>
TcpHybridMlRl::Fork()
{
    return CopyObject<TcpHybridMlRl>(this);
}

} // namespace ns3
