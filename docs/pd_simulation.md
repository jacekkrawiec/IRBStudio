# Problem statement

I want this package to answer following question: "What would be my portfolio's RWA if my PD model's accuracy went up to 80% from 75% (AUC)?"

# Data availability

I have full historical dataset with monthly snapshots on the facility level. I.e. there are as many observations per facility as many months it's on the book.
There is into_default_flag available, indicating if facility goes into default within 12 months after reporting date.
There is current model's output (PD score) available.
For the most recent part of the dataset (11 monthly snapshots) there is no full 12-months observation window available, so it is not known if these facilities go into default within 12 months or not.
The last monthly snapshot is an application sample.

# Considerations

[1] User wants to know RWA given changed AUC.
[2] There is no direct link between AUC and RWA.
[3] To simulate PD model with target AUC we would need to simulate PD scores, separately for 2 populations:
- defaulted
- non-defaulted
[4] We can assume certain characteristics of both populations, e.g. mean and variance. Assuming constant variance we can control the difference in means of both populations to control overlap of two scores distributions. This way AUC is directly achievable.
[5] This is fairly easy for historical population, where the outcome is known and it is feasible to divide population into 2 subpopulations, however this is not feasible directly for the application sample, where the split is not possible.
[6] Given that we know current model's output we can however fit mixture of beta distributions to current scores. Subsequently we can draw application sample scores from this distribution. We would use mixture beta assuming PD scores concetrate close to 0 as most of the book never defaults. Fitting this distributions to known scores aims to maintain similar scores' values to not distort pooling applied in next steps. 
[7] We need to remember facilities present in the application sample were mostly present in the historical dataset as well. It is unreasonable to assign scores completely randomly to this population, as it would cause unjustified volatility of RWAs over time. [we need to assume migrations over time and apply these migrations to application sample].
[8] Implying from above:
    1. application sample can be further divided into 2 subgroups
        - present in historical dataset
        - not present in the historical dataset (new faciltiies)
    2. For facilities already present in the historical dataset we can assign PD score based on historically observed migration matrices. 
    3. For faciltiies not present in the historical dataset we need to draw scores randomly from fitted distribution.

[9] Assuming this covers PD score simulation we need to proceed to the next step which is pooling and then calibration
[10] Pooling impacts the RWA calculations, we could run our own pooling algorithm, but it would blurry final RWA estimations. I.e. we would never know if the RWA impact comes from changes in ranking method (better AUC) or from different pooling strategy. To mitigate this we could use user's pooling as a baseline scenario. We need to ask user to provide not only scores but also ratings, so we can map scores to ratings or recreate these ratings from new scores. 
[11] As an alternative we can propose our own pooling and split RWA impact into 2 sources - impact coming from ranking and impact coming from pooling.
[12] As a final input to RWA formula we need calibrated PDs, so we need to calculate long-run average default rates in defined pools [long-run meaning average DR over time]
[13] As this process is naturally stochastic we can't have single RWA estimate, we need to run thousands of simulations to come up with expected RWA (average) with confidence intervals so can say expected RWA after model change lies between x and y.


# Implementation plan

Data Constraints & overall goal:
User will provide dataset with entire history of portfolio. It consists of monthly snapshots for each of facilities in the portfolio. E.g. n monthly snaphsots for facility present in the portfolio for n months. There will be already closed positions and still open ones. The target variable for PD model is into_default_flag, binary indicator if facility enters default in next 12 months. For majority of portfolio the target flag is known, for this part we can calculate AUC and generate scores reflecting new target AUC (this part of portfolio is called "historical sample). For most recent part of portfolio we don't know the target flag (facilities at reporting date for which full 12 months observation window is not available) - this part of portfolio is called an "application sample". We could ignore application sample in our considerations but ultimately we want to estimate simulated RWAs, which by nature are calculated on most recent data. Therefore we face following problem - how to generate PD scores for application sample, that will most probably achieve target AUC? Additionally, the application part of portfolio can consist of clients that are present in the historical sample - for them we cannot simply randomly draw PD score, therefore migration matrices are applied to avoid volatility. There is also a possibility to observe facilities entering portfolio during most recent 12 months, i.e. there is no history for them and the target flag is unknown - for them we will draw PD scores from distributions fitted to the "known" part of portfolio. We don't want to simulate any new facilities entering portfolio in the future - this will be kept as an additional post-MVP feature. For now we want to estimate RWAs in different AUC-driven scenarios on available data, i.e. for each reporting date present in the dataset, including most recent ones. We will predict feature including new faciltiies in the next features.

[1] Implement Beta mixture distribution fitter - mixes distributions of scores for defaulted and non-defaulted populations
    - Goal: fit beta mixture to observed PD scores to correctly reflect portfolio's risk profile.
[2] Historical migration matrices
    - Goal: define historical migration rates between ratings
    - Future use: apply migration matrices to "old clients" observations - i.e. those that were present also in the historical sample
[3] Score generator achieving target AUC on historical sample
    - Goal: generate PD scores matching target AUC on historical sample and reflecting expected risk profile on application sample
    - Future use: generate new set of PD scores on historical sample, as an input for pooling algo
    - Uses: Beta mixture (point 1), but with known class (i.e. defaulted/non-defaulted)
[4] Score generator for application sample - old clients
    - Goal: generate PD scores for application sample subset with old clients (present in historical sample)
    - Future use: generate new set of PD scores on app sample subset, as an input for pooling algo
    - Uses: Historical migration matrices (pont 2) - as migration matrices are based on ratings it needs to translate these ratings into scores (i.e. draw from score range within predicted rating)
[5] Score generator for application sample - new clients
    - Goal: generate PD scores for application sample subset with new clients (entered portfolio in last 11 months)
    - Future use: generate new set of PD scores on app sample subset, as an input for pooling algo
    - Uses: Beta mixture (point 1), but with unknown class (i.e. defaulted/non-defaulted), it must assume some weight of distributions, equal to expected default rate (to be provided by user or calculated in the package)
[6] Mapping to user-defined ratings
    - Goal: map new PD scores to user-defined ratings [can be provided by user or implied from data]
    - Future use: calibration
    - Uses: PD scores generated in points 3-5
[7] Pooling algorithm
    - Goal: define new pools/ratings, as an additional improvenet to the model; acts as an alternative 
    - Future use: calibration
    - Uses: PD scores genrated in points 3-5
[8] LRA DR calibration
    - Goal: recalculate long-term default rates for remapped ratings; returns 2 sets of LRAs, for user-defined ratings (point 6) and new ratings (point 7)
    - Future use: RWA estimation
    - Uses: ratings (points 6 and 7)
[9] RWA estimator
    - Goal: calculate final RWA based on simulated scores
    - Future use: reporting
    - Uses: calibrated PDs from point 8 + other user input (LGD, correlation etc.)
[10] Orchestrator/pipeline/monte carlo
    - Goal: repeat stochastic steps defined in 1-9 n thousand times to obtain confidence intervals around expected RWA
    - future use: reporting
    - uses: all steps 1-9