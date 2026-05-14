# Hotel-Booking-Cancellation-Prediction
Optimizing Revenue Management through Behavioral Forecasting.

📌 Business Context
In the hospitality sector, a "No-show" or a late cancellation is a direct loss of perishable inventory. This project predicts the likelihood of a booking being canceled, allowing hotels to:

Optimize Overbooking Ratios: Safely overbook rooms to ensure 100% occupancy.

Dynamic Deposit Policies: Require non-refundable deposits from "high-risk" behavioral profiles.

📊 Dataset Analysis
Using the Hotel Booking Demand dataset, I analyzed over 119k observations. Key features included:

Lead Time: The number of days between booking and arrival.

Deposit Type: Whether the customer paid upfront.

Previous Cancellations: Historical reliability of the guest.

🚀 Key Technical Decisions
High-Cardinality Management: Grouped 170+ countries into Top 10 + 'Other' to prevent overfitting and reduce feature space dimensions.

Advanced Boosting: Employed XGBoost with a low learning rate and early stopping to capture non-linear relationships between lead time and cancellation risk.

Model Explainability: Integrated SHAP to identify that deposit_type and lead_time are the strongest predictors of behavioral change.

📈 Results
ROC-AUC: ~0.90 (excellent separation of classes).

Recall (Cancellations): Prioritized identifying cancellations to help Revenue Managers mitigate risk.
