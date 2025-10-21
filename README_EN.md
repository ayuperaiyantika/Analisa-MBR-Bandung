# Analisa-MBR-Bandung

## Business Understanding
The Bandung City Government has a work program to provide social assistance to the poor. In order to make these funds effective and efficient to reach the right targets, the Bandung City Government needs valid data regarding the economic conditions of the community in each sub-district. This is certainly relevant to Indonesia's current condition (the presence of COVID-19), which requires the Bandung City Government to prioritize low-income communities.

Research Questions:
- How many residents are in a region based on sub-district in Bandung City in 2017-2020?
- Which areas have the highest and lowest population density?
- What is the correlation between MBR residents and poor residents with elementary school graduates in the sub-districts of Bandung City?
- Predict the number of poor residents based on the number of elementary school graduates

# Data Understanding
<img width="433" height="198" alt="image" src="https://github.com/user-attachments/assets/8f78c6bf-c265-443f-ac74-9716d306ab62" />
<img width="467" height="201" alt="image" src="https://github.com/user-attachments/assets/947eca90-0de8-482a-b287-99fe844ce4e3" />

# Exploratory Data Analysis
<img width="764" height="207" alt="image" src="https://github.com/user-attachments/assets/d11d552c-3d6a-4d6d-96d6-1f91aa803404" />

- Population Distribution
  <img width="596" height="450" alt="image" src="https://github.com/user-attachments/assets/ad0a44b6-8845-4e1b-b010-006a4b9975dd" />
- Correlation between MBR Residents and Poor Residents with Elementary School Graduates
  <img width="400" height="243" alt="image" src="https://github.com/user-attachments/assets/101c2a12-50fa-4e19-b1e4-6f744a8ea0a8" />

# Conclusion
1. This program can determine the number of residents in an area based on sub-districts in Bandung City, which is implemented in the population distribution graph in each sub-district (from 2017-2020)
2. The area with the highest population density is "Babakan Ciparay" sub-district, while the sub-district with low population density is "Cinambo"
3. The correlation results between MBR residents and poor residents with elementary school graduates in the sub-districts of Bandung City show a positive value with a Pearson correlation coefficient of 0.6942072305666471.
4. The prediction of the number of poor residents based on the number of elementary school graduates using the Lasso regression model that has been fine-tuned with GridSearchCV resulted in an r2 score of 0.6792875608228346 and an RMSE value of 1190.670238990295.