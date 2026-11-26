# Convener 질문에 따른 limit 및 background-fit 검증 기록

2026년 11월 24–26일의 NPS-26-009 blinded 검증을 정리했다. Production 기준 batch는 `full_spectrum_outputs/condor/AN_runCorr_20261125_051941`이다. 아래 mass의 단위는 GeV이다.

| 확인하려던 질문 | 필요한 계산 | 확인한 결과 |
|---|---|---|
| Sideband가 background normalisation을 constrain할 때 expected limit은 얼마인가? | Run-correlated full-spectrum model의 background-only Asimov expected limit | 57/57 mass point 완료. M12 한 점의 limit에 약 2시간 29분 소요 |
| Fit과 uncertainty 계산이 안정적인가? | Representative fits, likelihood scans, nuisance impacts, signal-injection toys | Scan crossing 확인. 일부 covariance 및 toy fit 실패가 있어 job 종료 여부와 fit 품질을 별도로 판정 |
| M70에서 asymptotic approximation은 어떤가? | 동일 mass의 median expected HybridNew 및 asymptotic limit 비교 | HybridNew가 약 3.2% 낮음 |
| DY yield가 예상과 ±20% 다르면 fit이 얼마나 회복하는가? | DY만 바꾼 pseudo-data에 nominal model로 background-only fit | 모든 tested mass와 두 Run에서 injected DY가 postfit interval 안에 있음. QCD와 ttbar central value의 변화는 각각 최대 약 4.3%, 2.0% |

## 1. Production model과 binning

기존 window counting에서 11–80 GeV dimuon spectrum을 함께 fit하는 방식으로 옮겨, sideband가 background normalisation을 constrain하도록 했다. 각 period의 mass resolution으로 정한 ±5σ window 경계는 0.02 GeV 단위로 바깥쪽으로 맞추고, 나머지 spectrum에는 central sideband binning을 적용했다. QCD, DY, ttbar에는 각각 Run 2와 Run 3에 하나씩, 총 6개의 normalisation nuisance를 사용했다. 각 nuisance는 같은 Run의 period들이 공유하고 두 Run 사이에는 독립이며, prefit log-normal factor는 1.5이다. Central transfer factor의 Run별 산출과 nuisance correlation 설정은 각각 기록해야 한다.

M12, M30, M70에서 coarse/central/fine sideband binning을 비교한 이전 테스트에서는 median expected limit 차이가 최대 약 0.65%였다. 이 수치는 당시 correlation 설정에서 얻었다. Postfit distribution 표시용 fixed-20-bin, no-window fit도 따로 수행했으므로 production limit을 재현할 때는 production batch의 binning과 datacard를 사용한다.

## 2. Expected limits와 representative fits

`limits` batch에서는 `--fit-components none --limits`로 별도의 FitDiagnostics 작업 없이 blinded AsymptoticLimits를 수행했다. Minimiser strategy 1, rMin 0, 초기 rMax 10, adaptive upper cap 100을 사용했다. `representative` batch는 M12, 20, 30, 40, 50, 60, 70에서 별도로 fit 품질을 조사했다. 아래 시간은 batch status의 job별 경과시간이며 queue 대기시간이나 전체 DAG 소요시간으로 사용할 수 없다.

| Mass | Median expected r | Median expected α | Limit elapsed | Background-only fit elapsed | Signal-plus-background fit elapsed |
|---:|---:|---:|---:|---:|---:|
| 12 | 9.28125 | 1.87813 × 10⁻⁶ | 8954.8 s (149.2 min) | 296.2 s | 2500.1 s |
| 40 | 1.7734375 | 3.27981 × 10⁻⁵ | 3948.9 s (65.8 min) | 391.8 s | 2981.2 s |
| 70 | 0.890625 | 3.67481 × 10⁻⁴ | 3612.2 s (60.2 min) | 366.2 s | 2567.6 s |

M40 workspace 생성은 68.4 s였다. 다음 submission에서 비슷한 model의 background-only fit은 대표점 실측 5–7분, signal-plus-background fit은 42–50분, limit은 60–150분을 첫 추정치로 쓸 수 있다. Slot 대기, node별 CPU 성능, rMax retry는 별도로 고려한다.

57개 limit은 모두 완료됐고 representative signal-plus-background fit도 7/7 완료됐다. 반면 M50 background-only fit은 status 0이어도 covQual 2여서 품질 기준에서 탈락했다. Resume DAG 19744329가 제출됐으며 최종 M50 covariance는 이 기록에서 확인되지 않았다. Limit 결과와 해당 FitDiagnostics covariance의 유효성은 따로 판정한다.

## 3. Likelihood, impacts, toys

| 검증 | 문제와 실행 방법 | 결과와 해석 |
|---|---|---|
| Profile likelihood | M20, M70에서 signed r을 허용한 241-point scan으로 r=0 양쪽의 profile을 조사 | 두 mass 모두 −2ΔlnL 기준 1과 4의 crossing을 확보했다. Best-fit r은 각각 약 −4.05 × 10⁻⁸, +6.38 × 10⁻⁸. 원본의 point별 minimiser status는 확인되지 않았다. |
| Nuisance impacts | M12, M40, M70에서 r=0, 1로 여섯 panel을 만들었다. 일부 nuisance crossing이 실패해 성공한 output을 재사용하고 M12 r=1 및 M40 r=0, 1의 실패 node를 대상으로 repair | Impact를 판단할 때 nominal fit뿐 아니라 nuisance별 up/down crossing을 검사한다. 해당 repair 대상은 J1503, J1738, J2682였다. |
| Signal-injection toys | M12, M30, M70 각각에 r=0, 해당 mass의 expected r95, 2r95를 주입해 조합당 100개, 총 900개 수행 | 최초 883/900 fit 성공. M70 r=0에서 성공한 toy들의 평균 r은 0보다 평균의 standard error 약 3배만큼 낮았다. 실패를 포함한 무조건부 bias나 interval coverage로 해석할 수 없다. |
| Toy refit | 실패 entry가 들어 있는 chunk를 다시 fit하고 seed·entry 단위로 원본과 대조 | 재실행한 80개 중 70개 성공. 기존 실패 13개가 회복됐고 기존 성공 6개가 새로 실패했다. 재시도 결과로 chunk 전체를 덮으면 기존 성공도 잃을 수 있다. |
| M70 HybridNew | 29개 grid point에 각각 500 toys를 한 job에 넣었을 때 일부 실행 중 job의 누적 CPU가 이미 5951–5985 s였다. Grid point당 50 toys씩 10개 job으로 나누도록 변경 | 같은 총 toy 수에서 290개 grid job을 제출할 수 있다. M70 median expected r은 HybridNew 0.86208, asymptotic 0.890625 (ratio 0.96795). 한 mass에서 두 방법의 expected limit을 비교한 결과다. |

## 4. DY heavy-flavour 질문과 injection recovery (AN Appendix J.4)

Matteo의 질문은 b-tagging selection에 따른 heavy-flavour composition 차이가 DY yield를 바꿀 때 기존 fit이 그 변화를 얼마나 회복하는지였다. 처음 제출한 `norm_correlation` DAG 19778498은 nominal pseudo-data를 고정하고 era/run/global correlation만 바꿨다. 3 mass × 3 scope에 workspace, background-only fit, limit을 각각 생성하고 마지막에 collect해 총 28개 node였다. Injected DY가 1이므로 postfit central value가 1인 결과로는 ±20% recovery를 판단할 수 없었다. 전체 collector는 마지막 era limit까지 기다렸고, Run-only 결과는 `norm_correlation_run_only_r0ezicjg`로 별도 수집했다. 해당 스크립트는 삭제하고 DY injection workflow로 교체했다.

새 workflow는 각 bin의 background-only pseudo-observation을 `B_nom + (k − 1) × DY_nom`으로 구성한다. 두 Run에 같은 k를 적용하고 QCD와 ttbar의 생성 yield는 고정한다. Fit에는 기존 Run-correlated model과 nominal nuisance prior를 그대로 사용한다. Fit 단계에서 nominal Asimov data를 다시 생성하면 주입한 변화가 사라지므로 저장한 pseudo-observation을 읽는다.

아래 값은 fitted normalisation scale factor이다. DY에는 postfit [low, high]를 함께 표시했다. QCD와 ttbar의 interval까지 포함한 원본은 batch의 `postfit_normalisations.csv`에 있다.

| Mass | Injected DY | Run 2 DY [low, high] | Run 2 QCD | Run 2 ttbar | Run 3 DY [low, high] | Run 3 QCD | Run 3 ttbar |
|---:|---:|---|---:|---:|---|---:|---:|
| 12 | 0.8 | 0.834 [0.706, 0.986] | 0.963 | 0.982 | 0.874 [0.687, 1.111] | 0.958 | 0.982 |
| 12 | 1.0 | 1.000 [0.863, 1.158] | 1.000 | 1.000 | 1.000 [0.795, 1.258] | 1.000 | 1.000 |
| 12 | 1.2 | 1.175 [1.027, 1.346] | 1.026 | 1.013 | 1.141 [0.916, 1.420] | 1.035 | 1.014 |
| 40 | 0.8 | 0.835 [0.706, 0.988] | 0.963 | 0.981 | 0.874 [0.687, 1.112] | 0.957 | 0.982 |
| 40 | 1.0 | 1.000 [0.878, 1.139] | 1.000 | 1.000 | 1.000 [0.795, 1.258] | 1.000 | 1.000 |
| 40 | 1.2 | 1.175 [1.025, 1.347] | 1.026 | 1.013 | 1.141 [0.916, 1.421] | 1.035 | 1.014 |
| 70 | 0.8 | 0.836 [0.706, 0.991] | 0.962 | 0.980 | 0.875 [0.687, 1.115] | 0.957 | 0.982 |
| 70 | 1.0 | 1.000 [0.860, 1.163] | 1.000 | 1.000 | 1.000 [0.793, 1.260] | 1.000 | 1.000 |
| 70 | 1.2 | 1.174 [1.022, 1.348] | 1.027 | 1.014 | 1.139 [0.913, 1.422] | 1.035 | 1.015 |

9개 fit 모두 status 0, covQual 3이었다. Nominal injection에서는 세 background 모두 unity로 돌아왔다. ±20% injection에서 central shift recovery를 (fitted DY − 1)/(k − 1)로 계산하면 Run 2 약 82–88%, Run 3 약 62–70%이다. Run 3의 central value는 nominal 쪽에 더 가깝고 interval도 넓다. 각 interval은 injected value를 포함한다. QCD와 ttbar의 몇 % 변화는 각 process 자체의 nominal yield 대비 값이다. 실제 이동한 event 수를 비교하려면 process별 nominal yield를 곱해야 한다.

### 동일한 DY test 제출

`NIsoMuon`에서 실행한다. 이미 완료한 batch는 status와 CSV만 읽고, 새 submission에는 새 batch 경로를 지정한다.

```bash
source_batch=full_spectrum_outputs/condor/AN_runCorr_20261125_051941/representative
batch=full_spectrum_outputs/condor/dy_injection_recovery
python3 dy_injection_recovery.py submit \
  --source "$source_batch" --batch "$batch" --masses 12,40,70 --no-limits
python3 dy_injection_recovery.py status --batch "$batch"
column -s, -t < "$batch/postfit_normalisations.csv"
```

DAG 19779248은 9개 독립적인 workspace → fit chain과 마지막 collector 1개, 총 19개 node였다. 최대 9개 chain이 동시에 진행될 수 있다. `--no-limits`에서는 `expected_limits.csv`가 생성되지 않는다. Limit comparison은 DY mismodelling이 signal sensitivity에 미치는 영향까지 질문받았을 때 추가한다.

## 5. Memory와 다음 submission에서 남길 기록

| 작업 | 확보된 resource 정보 | 판단할 때 주의할 점 |
|---|---|---|
| Production / representative | Representative workspace request 2048 MB. 당시 production fit/limit에서 논의된 request 1024 MB. 이전 로그의 RSS 관측치 약 1.15 GiB | RSS 관측치를 특정 mass·stage에 연결할 근거가 부족하다. M50 covariance 문제를 OOM으로 진단할 근거도 없다. |
| DY injection | 스크립트 기본 RequestMemory는 workspace 2048 MB, background-only fit 4096 MB, collector 및 optional limit 2048 MB | Request는 scheduler 예약량이다. 이 DAG의 job별 실측 peak memory와 elapsed time은 보존된 출력에서 확인되지 않았다. |
| Correlation 비교 | 일부 1-CPU limit job의 RemoteUserCpu가 약 600 → 1500 → 4190 s로 증가 | 실행 중 CPU 누적 snapshot이다. 최종 wall time 또는 DY injection runtime 추정에 합산하지 않는다. |
| Scans, impacts, toys, HybridNew | 위에 기재한 running-job CPU snapshot 외에 일관된 stage별 최종 elapsed/RSS 표는 확보하지 못했다 | 다음 실행 때 DAG별 history와 event log를 collect 결과에 보관한다. |

1. **질문별 최소 DAG를 만든다.** Normalisation recovery에는 세 injection과 background-only fit이면 된다. Fit summary를 optional limit 뒤에 연결하면 마지막 slow limit 하나가 결과 확인을 지연시킨다. 각 collector는 필요한 parent에만 의존시킨다.
2. **Production 설정을 고정한다.** Git SHA, source batch, datacard/workspace 경로, mass, binning, nuisance correlation, injection truth와 fit options를 manifest에 남긴다. DY yield scale k와 signal strength r을 별도 필드로 둔다.
3. **품질 검사를 output 종류별로 둔다.** Fit status, covQual, EDM과 boundary를 읽고, scan·impact는 crossing, limit은 quantile 순서와 scan range, toy는 requested/recorded/successful 수를 검사한다. Nominal DY injection에서는 unity closure, shifted injection에서는 interval과 process별 central shift를 함께 확인한다.
4. **비싼 계산은 작은 단위로 분할한다.** Mass·injection·grid point를 병렬화하고 긴 toy job은 독립 seed를 가진 chunk로 나눈다. Merge 시 mass, grid point, seed, toy 수, 누락과 중복을 검사한다. Repair는 원본을 보존하고 실패한 entry와 변경된 status를 기록한다.
5. **Pilot에서 request와 실제 사용량을 함께 측정한다.** 대표 low/mid/high mass에서 먼저 돌리고 wall time과 memory의 큰 쪽을 기준으로 stage별 request를 정한다. 실행 중에는 `condor_tail`로 worker stdout/stderr를 확인하고, 종료 후에는 아래 명령으로 job별 history를 보관한다. 재시작한 job은 event log도 함께 확인한다.

```bash
dag=19779248  # 새 실행에서는 제출된 DAGMan cluster ID를 사용
condor_q -constraint "DAGManJobId == $dag" \
  -af:h ClusterId ProcId DAGNodeName JobStatus RemoteUserCpu HoldReason
condor_history -constraint "DAGManJobId == $dag" \
  -af:h ClusterId ProcId DAGNodeName RequestMemory MemoryUsage ResidentSetSize \
  RemoteWallClockTime RemoteUserCpu NumJobStarts ExitCode
```

RequestMemory와 MemoryUsage는 MB, ResidentSetSize는 KiB, RemoteWallClockTime과 RemoteUserCpu는 초 단위로 기록한다. DAG 전체 turnaround는 submission부터 마지막 필수 collector 완료까지 별도로 측정한다.

## 근거 자료

- [Production, likelihood, impact, toy 및 HybridNew 결과 archive](https://github.com/joonblee/higgs_combine/tree/0b0c3c95549c10fe1d6be67a84f886d1d758bf4c/NIsoMuon/results/AN_runCorr_20261125_051941/values)
- [HybridNew grid point별 toy 분할 변경](https://github.com/joonblee/higgs_combine/commit/9c081268)
- [DY injection 구현 및 기존 correlation study 교체](https://github.com/joonblee/higgs_combine/commit/debb9b80c01409af3e25797f6a89acc7d0e6e556)
- [AN Appendix J.4 변경](https://github.com/joonblee/CMS-NPS-26-009_AN/commit/75d03c4da657c7f2e63defe4149c3a92ec444ef3)
- 11월 25–26일 tamsa2 출력에서 확인한 측정값은 production status의 `[TIME]` 행, DAG 19778498의 `condor_q` snapshot, DAG 19779248의 status 및 `postfit_normalisations.csv`를 기준으로 했다.
