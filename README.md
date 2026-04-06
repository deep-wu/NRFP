
## Dataset Preparation Guideline
Please prepare the datasets according to the configuration files in the data folder. You may refer to these sites to download the datasets: [Office-31 / Office-Home](https://github.com/tim-learn/SHOT), [DomainNet-126](https://ai.bu.edu/M3SDA/). The default data structure is given as follows:

```
./data
|–– office/
|–– domainnet-126/
|–– office-home/
|   |–– domain1/
|   |–– domain2/
|   |-- ...
|   |–– domain1.txt
|   |-- domain2.txt
|   |-- ...
```

## Training Scripts
Training scripts are provided for each dataset separately (office31.sh, home.sh, domainnet.sh). To train the model, simply execute the appropriate script. For example, to run the experiment on DomainNet-126, use the following command:
    ```
    sh domainnet.sh
    ```
    
