#ifndef TCPHYBRIDMLRL_H
#define TCPHYBRIDMLRL_H

#include "tcp-congestion-ops.h"

#include <memory>
#include <string>

namespace ns3
{

class TcpHybridMlRl : public TcpCongestionOps
{
  private:
    struct ModelBundle;

  public:
    static TypeId GetTypeId();

    TcpHybridMlRl();
    TcpHybridMlRl(const TcpHybridMlRl& other);
    ~TcpHybridMlRl() override;

    std::string GetName() const override;

    void PktsAcked(Ptr<TcpSocketState> tcb,
                   uint32_t segmentsAcked,
                   const Time& rtt) override;

    void CwndEvent(Ptr<TcpSocketState> tcb,
                   const TcpSocketState::TcpCAEvent_t event) override;

    void CongestionStateSet(
        Ptr<TcpSocketState> tcb,
        const TcpSocketState::TcpCongState_t newState) override;

    void IncreaseWindow(Ptr<TcpSocketState> tcb,
                        uint32_t segmentsAcked) override;

    uint32_t GetSsThresh(Ptr<const TcpSocketState> tcb,
                         uint32_t bytesInFlight) override;

    Ptr<TcpCongestionOps> Fork() override;

  private:
    Time m_latestRtt;
    Time m_minRtt;
    uint32_t m_lastCwnd;
    uint32_t m_lastLossEvent;

    // 0 = Low, 1 = Medium, 2 = High
    uint32_t m_mlCongestionState;

    // Shared ONNX sessions remain valid when ns-3 forks the controller.
    std::shared_ptr<ModelBundle> m_models;
    std::string m_modelDirectory;
    std::string m_decisionLogPath;
    double m_linkBandwidthMbps;
    double m_linkDelayMs;
    double m_lastDecisionTime;
    double m_lastThroughputMbps;
    uint64_t m_totalAckedBytes;
    uint64_t m_lastDecisionAckedBytes;
    double m_cwndFactor;
    uint64_t m_lastLossCount;
    uint32_t m_lastPpoAction;
};

} // namespace ns3

#endif // TCPHYBRIDMLRL_H
