// The false-table check's weights (GridRule::check), generated: do not edit. Fitted on labelled pages
// in no benchmark (DocLayNet's and PubTables-1M's validation pages); a logistic over table_features'
// features, each standardised by its mean and scale.
#pragma once

namespace carbon::detail {

inline constexpr int kTableCheckVersion = 1;
inline constexpr int kTableFeatures = 26;

struct TableCheckWeights {
  float mean[kTableFeatures], scale[kTableFeatures], weight[kTableFeatures], bias;
};

// pages read by OCR: 3104 tables (2167 true); on the held-out folds 2144 of 2167 true
// tables kept and 740 of 937 false ones written as text
inline constexpr TableCheckWeights kTableCheckOcr{
    {1.6798f, 2.23761f, 0.818139f, 0.766849f, 0.0795747f, 2.07394f, 0.392783f, 0.289404f,
     0.00444868f, 1.03109f, 0.0128897f, 0.101871f, 3.34119f, 0.58661f, 0.0836563f, 6.91069f,
     0.0688632f, 0.90615f, 0.0433733f, 6.4433e-10f, 6.4433e-10f, 0.0112739f, 0.240963f, 0.0943779f,
     0.168343f, 0.362823f},
    {0.479436f, 0.829941f, 0.170477f, 0.311119f, 0.270634f, 0.562056f, 0.297473f, 0.212044f,
     0.0338285f, 0.3002f, 0.0296562f, 0.176713f, 1.1958f, 0.25158f, 0.0612092f, 8.13254f,
     0.120138f, 0.160303f, 0.0853208f, 1.0f, 1.0f, 0.0396747f, 0.260405f, 0.228556f,
     0.354005f, 0.269765f},
    {-1.08796f, 0.237378f, -0.226741f, 0.522229f, 0.210526f, -0.0039828f, -0.0986429f, -0.147202f,
     -0.465799f, -0.229007f, -0.355545f, -0.292858f, 2.22301f, 0.498929f, -0.729065f, 0.42749f,
     -0.0793594f, 1.19582f, -0.0472463f, -1.90913e-08f, -1.90913e-08f, 0.25374f, 1.14847f, -0.397291f,
     -2.7382f, -0.475544f},
    3.56531f};

// pages read from their text layer: 976 tables (486 true); on the held-out folds 476 of 486 true
// tables kept and 286 of 490 false ones written as text
inline constexpr TableCheckWeights kTableCheckTextLayer{
    {1.56388f, 2.14472f, 0.766693f, 0.658935f, 0.112705f, 2.04994f, 0.334257f, 0.23763f,
     0.00702507f, 1.09864f, 0.0220322f, 0.105318f, 3.01271f, 0.562464f, 0.111902f, 8.11452f,
     0.0979068f, 0.846437f, 0.079957f, 2.04918e-09f, 2.04918e-09f, 0.0208974f, 0.219485f, 0.198165f,
     0.099618f, 0.379073f},
    {0.52094f, 0.91092f, 0.205821f, 0.338183f, 0.316232f, 0.677667f, 0.306529f, 0.226447f,
     0.0354842f, 0.366585f, 0.0672025f, 0.178893f, 1.23874f, 0.267836f, 0.193669f, 10.188f,
     0.153686f, 0.19187f, 0.112945f, 1.0f, 1.0f, 0.055725f, 0.287858f, 0.348439f,
     0.286756f, 0.316104f},
    {-0.246012f, 0.300962f, 0.236909f, 0.510901f, 0.169181f, 0.0990143f, 0.44622f, -0.299744f,
     -0.13064f, 0.140903f, -0.733851f, -0.162076f, 0.565254f, 0.853406f, -0.853359f, -0.0766457f,
     -0.394378f, 0.727957f, -0.00784957f, -3.24938e-09f, -3.24938e-09f, 0.0852925f, 0.751359f, -0.0892783f,
     -0.633782f, 0.111506f},
    1.5792f};

}  // namespace carbon::detail
