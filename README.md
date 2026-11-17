# DSBA-6211-Airbnb-Project
EDA explaination

The target variable monthly_revenue is derived directly from the raw estimated_revenue_l365d column (InsideAirbnb's estimated total revenue over the last 365 days) divided by 12.0
  -estimated_revenue_l365d is present for 9,515 of the 10,242 listings (the remaining 727 rows were dropped)
  -for the active listings dataset (n=7,679) the modeling target is log_monthly_revenue
  -estimated_occupancy_l365d was dropped so there is no target leakage
      -the airbnb data used occupancy to derive estimated revenue, OUR code only uses estimated_revenue_l365d to derive our target variable monthly_revenue, so we drop both to avoid target leakage
  
Starting from the raw dataset (10,242 rows  90 columns), we removed 727 rows that were missing the target variable (monthly_revenue), dropped 2 target-leakage columns (estimated_revenue_l365d and estimated_occupancy_l365d), dropped 14 columns missing >70% of values, and imputed remaining missing values using mean (|skew| <= 0.5) or median (|skew| > 0.5).

Because 1,836 listings (19.3%) had $0 in estimated monthly revenue (1,162 brand-new/unreviewed listings with 0 reviews + 674 inactive listings with past reviews), keeping them in a single regression target creates a massive $0 spike that distorts log(monthly_revenue). To handle this without throwing away data or skewing our target distribution, we exported two clean datasets using a two part setup:

  -cleaned_all_listings (9,515 rows, 92 columns)
    -Contains all valid listings (both active and $0-revenue listings), with reviews_per_month set to 0.0 for unreviewed listings and two binary flags added: is_active_revenue (1 if monthly_revenue > 0, 0 otherwise) and has_reviews (1 if number_of_reviews > 0, 0 otherwise).
        - we should use this for classification models predicting wether a listing generates revenue at all (is_active_revenue) or full sample baseline comparisons

  -cleaned_active_listings (7,679 rows, 93 columns) --> (log_monthly_revenue) included in this data
    -Filtered strictly to revenue-generating listings (monthly_revenue > 0, min $15.25/mo), where log_monthly_revenue = np.log(monthly_revenue) is clean, continuous, and unimodal (mean = 7.62, median = 7.86, skew = -0.78), with zero imputed review scores.
          -we should use this for regression models (OLS, ridge/lasso, random forest) predicting log_monthy_revenue (** need to make sure we convert predictions back into $ when comparing MAE/RMSE**)

EXTRA NOTES:
-linear models will benefit from log-transforming skewed predictors 

-The 9,515 listings belong to only 3,569 unique hosts (host_id), and commercial hosts manage up to 4,033 properties. If your team uses a standard random train_test_split, listings from the exact same host (or same apartment building) can appear in both train and test sets. For stricter evaluation, consider GroupShuffleSplit(groups=df['host_id']) or at minimum lock in the exact same train_test_split(..., test_size=0.2, random_state=42) indices so every model is scored on the exact same test set.
