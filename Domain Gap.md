# Baseline
## City
### Boston -> Boston
Evaluating bboxes of img_bbox
                                                                                                    mAP: 0.2872                                                                                               
mATE: 0.7034
mASE: 0.2933
mAOE: 0.7837
mAVE: 1.1701
mAAE: 0.3729
NDS: 0.3282
Eval time: 92.0s

Per-class results:
Object Class    AP      ATE     ASE     AOE     AVE     AAE
car     0.464   0.565   0.173   0.362   1.679   0.386
truck   0.217   0.772   0.219   0.351   1.438   0.332
bus     0.320   0.751   0.203   0.537   2.875   0.591
trailer 0.154   1.167   0.232   0.777   0.630   0.273
construction_vehicle    0.065   0.925   0.540   1.360   0.108   0.297
pedestrian      0.206   0.778   0.309   1.514   1.003   0.814
motorcycle      0.247   0.596   0.314   0.945   0.602   0.039
bicycle 0.286   0.414   0.291   1.018   1.025   0.253
traffic_cone    0.463   0.553   0.364   nan     nan     nan
barrier 0.450   0.514   0.288   0.189   nan     nan
--------------------------------------------------------------------------------
### Boston -> Singapore
Evaluating bboxes of img_bbox
                                                                                                        mAP: 0.1257                                                                                               
mATE: 0.8702
mASE: 0.4048
mAOE: 1.0747
mAVE: 1.0305
mAAE: 0.5414
NDS: 0.1812
Eval time: 52.1s

Per-class results:
Object Class    AP      ATE     ASE     AOE     AVE     AAE
car     0.281   0.752   0.191   0.525   2.392   0.504
truck   0.051   0.889   0.325   0.687   1.152   0.417
bus     0.140   0.918   0.222   1.018   1.412   0.509
trailer 0.000   1.000   1.000   1.000   1.000   1.000
construction_vehicle    0.007   1.118   0.604   1.625   0.091   0.782
pedestrian      0.207   0.850   0.321   1.507   0.711   0.656
motorcycle      0.088   0.796   0.291   1.527   1.424   0.463
bicycle 0.012   0.857   0.307   1.404   0.061   0.000
traffic_cone    0.225   0.787   0.412   nan     nan     nan
barrier 0.245   0.735   0.374   0.380   nan     nan
--------------------------------------------------------------------------------
### Boston -> Singapore fine-tune
Evaluating bboxes of img_bbox
                                                                                                                                                       mAP: 0.2523                                                                                                                                              
mATE: 0.7492
mASE: 0.3574
mAOE: 0.8310
mAVE: 1.2845
mAAE: 0.5010
NDS: 0.2823
Eval time: 55.6s

Per-class results:
Object Class    AP      ATE     ASE     AOE     AVE     AAE
car     0.420   0.620   0.166   0.344   2.189   0.306
truck   0.201   0.674   0.212   0.341   1.686   0.516
bus     0.321   0.716   0.200   0.278   2.546   0.581
trailer 0.000   1.000   1.000   1.000   1.000   1.000
construction_vehicle    0.043   1.044   0.601   1.667   0.117   0.659
pedestrian      0.266   0.802   0.306   1.441   0.710   0.675
motorcycle      0.270   0.735   0.255   1.060   1.950   0.267
bicycle 0.232   0.659   0.272   1.027   0.078   0.004
traffic_cone    0.305   0.644   0.283   nan     nan     nan
barrier 0.466   0.597   0.279   0.321   nan     nan
--------------------------------------------------------------------------------
## Lighting 
### Day -> Day
Evaluating bboxes of img_bbox
                                                                                                        mAP: 0.2694                                                                                               
mATE: 0.7266
mASE: 0.2863
mAOE: 0.9024
mAVE: 1.2275
mAAE: 0.4138
NDS: 0.3018
Eval time: 140.9s

Per-class results:
Object Class    AP      ATE     ASE     AOE     AVE     AAE
car     0.435   0.600   0.172   0.410   1.789   0.372
truck   0.187   0.807   0.234   0.406   1.539   0.381
bus     0.294   0.710   0.213   0.479   2.796   0.589
trailer 0.115   0.983   0.233   0.937   0.623   0.298
construction_vehicle    0.047   0.965   0.523   1.348   0.110   0.415
pedestrian      0.238   0.792   0.305   1.487   0.828   0.730
motorcycle      0.300   0.696   0.262   1.411   1.518   0.381
bicycle 0.250   0.558   0.269   1.349   0.617   0.144
traffic_cone    0.396   0.580   0.358   nan     nan     nan
barrier 0.433   0.574   0.295   0.295   nan     nan
--------------------------------------------------------------------------------
### Day -> Night
Evaluating bboxes of img_bbox
                                                                                                        mAP: 0.1262                                                                                               
mATE: 0.8447
mASE: 0.4801
mAOE: 0.9263
mAVE: 1.7973
mAAE: 0.7654
NDS: 0.1614
Eval time: 4.8s

Per-class results:
Object Class    AP      ATE     ASE     AOE     AVE     AAE
car     0.430   0.567   0.153   0.433   4.149   0.758
truck   0.138   0.754   0.243   0.553   2.028   0.914
bus     0.000   1.000   1.000   1.000   1.000   1.000
trailer 0.000   1.000   1.000   1.000   1.000   1.000
construction_vehicle    0.000   1.000   1.000   1.000   1.000   1.000
pedestrian      0.115   0.704   0.269   1.347   0.757   0.531
motorcycle      0.176   0.710   0.269   1.529   4.274   0.920
bicycle 0.096   0.781   0.261   0.830   0.171   0.000
traffic_cone    0.000   1.229   0.338   nan     nan     nan
barrier 0.308   0.702   0.268   0.645   nan     nan
--------------------------------------------------------------------------------
## Weather
### Sunny -> Sunny
Evaluating bboxes of img_bbox
                                                                                                                                                       mAP: 0.2846                                                                                                                                              
mATE: 0.7145
mASE: 0.2745
mAOE: 0.7056
mAVE: 1.2439
mAAE: 0.4065
NDS: 0.3322
Eval time: 121.0s

Per-class results:
Object Class    AP      ATE     ASE     AOE     AVE     AAE
car     0.460   0.568   0.168   0.289   1.964   0.370
truck   0.221   0.771   0.226   0.301   1.613   0.387
bus     0.315   0.704   0.202   0.265   2.601   0.542
trailer 0.122   1.015   0.197   0.509   0.604   0.250
construction_vehicle    0.070   0.998   0.509   1.278   0.110   0.422
pedestrian      0.255   0.785   0.300   1.466   0.821   0.727
motorcycle      0.293   0.677   0.256   1.044   1.659   0.421
bicycle 0.254   0.549   0.256   0.959   0.579   0.133
traffic_cone    0.418   0.541   0.357   nan     nan     nan
barrier 0.439   0.537   0.276   0.239   nan     nan
--------------------------------------------------------------------------------
### Sunny -> Rainy
Evaluating bboxes of img_bbox
                                                                                                                                                       mAP: 0.2005                                                                                                                                              
mATE: 0.8611
mASE: 0.3052
mAOE: 0.9420
mAVE: 1.0464
mAAE: 0.3071
NDS: 0.2587
Eval time: 21.2s

Per-class results:
Object Class    AP      ATE     ASE     AOE     AVE     AAE
car     0.421   0.649   0.172   0.337   1.203   0.284
truck   0.161   0.858   0.218   0.407   0.952   0.275
bus     0.184   0.950   0.202   0.772   2.520   0.587
trailer 0.076   1.089   0.313   1.198   0.864   0.090
construction_vehicle    0.011   1.098   0.457   1.598   0.087   0.212
pedestrian      0.104   0.808   0.359   1.527   1.076   0.745
motorcycle      0.082   0.908   0.264   1.263   1.343   0.202
bicycle 0.180   0.957   0.366   1.121   0.327   0.062
traffic_cone    0.296   0.685   0.401   nan     nan     nan
barrier 0.489   0.610   0.301   0.255   nan     nan
--------------------------------------------------------------------------------
# BEVUDA-adv loss
