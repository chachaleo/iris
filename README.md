# Iris recognition

<div align="center">
  <img src="img/iris.png" alt="iris-logo" height="260"/>
</div>

Implementation of ZKML iris recognition, using the worldcoin iris recognition pipeline.

## Pipeline Steps and Corresponding Functions

1. **Segmentation** → `iris.MultilabelSegmentation.create_from_hugging_face` 
2. **Segmentation Binarization** → `iris.MultilabelSegmentationBinarization` 
3. **Vectorization** → `iris.ContouringAlgorithm` 
4. **Specular Reflection Detection** → `iris.SpecularReflectionDetection` 
5. **Interpolation** → `iris.ContourInterpolation` 
6. **Distance Filter** → `iris.ContourPointNoiseEyeballDistanceFilter` 
7. **Eye Orientation** → `iris.MomentOfArea` 
8. **Eye Center Estimation** → `iris.BisectorsMethod` 
9. **Eye Centers Inside Image Validator** → `iris.nodes.validators.cross_object_validators.EyeCentersInsideImageValidator` 
10. **Smoothing** → `iris.Smoothing` 
11. **Geometry Estimation** → `iris.FusionExtrapolation` 
12. **Pupil to Iris Property Estimation** → `iris.PupilIrisPropertyCalculator` 
13. **Offgaze Estimation** → `iris.EccentricityOffgazeEstimation` 
14. **Occlusion 90 Calculator** → `iris.OcclusionCalculator` 
15. **Occlusion 30 Calculator** → `iris.OcclusionCalculator` 
16. **Noise Masks Aggregation** → `iris.NoiseMaskUnion` 
17. **Normalization** → `iris.LinearNormalization` 
18. **Sharpness Estimation** → `iris.SharpnessEstimation` 
19. **Filter Bank** → `iris.ConvFilterBank` 
20. **Iris Response Refinement** → `iris.nodes.iris_response_refinement.fragile_bits_refinement.FragileBitRefinement` 
21. **Encoder** → `iris.IrisEncoder` 
22. **Bounding Box Estimation** → `iris.IrisBBoxCalculator`


## Pipeline Reduced

1. **Segmentation** → `iris.MultilabelSegmentation.create_from_hugging_face` ✅
2. **Segmentation Binarization** → `iris.MultilabelSegmentationBinarization` ✅
3. **Vectorization** → `iris.ContouringAlgorithm` ✅
4. **Distance Filter** → `iris.ContourPointNoiseEyeballDistanceFilter` ✅
5. **Eye Orientation** → `iris.MomentOfArea` ✅
6. **Eye Center Estimation** → `iris.BisectorsMethod` ✅
7. **Geometry Estimation** → `iris.FusionExtrapolation` ✅
8. **Normalization** → `iris.LinearNormalization` ✅
9. **Filter Bank** → `iris.ConvFilterBank` ✅
10. **Encoder** → `iris.IrisEncoder` ✅

-> The reduction does result to a lost of precision
-> The steps skipped are not the biggest ones in term of computation 

