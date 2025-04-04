# BPFragmentODRL Evaluation Report

## Overview

This report summarizes the evaluation of the BPFragmentODRL system, which implements automated generation of fragment-level policies for business processes using ODRL.

**Evaluation Date:** 2025-04-04 19:10:29

## Dataset Summary

- **Total Models:** 10
- **Successfully Processed Models:** 10
- **Failed Models:** 0
- **Average Activities per Model:** 8.20

## Fragmentation Results

- **Fragmentation Strategy:** gateway
- **Average Fragments per Model:** 6.80

## Policy Generation Results

- **Average Policy Generation Time:** 0.00 seconds
- **Average Permissions per Model:** 7.90
- **Average Prohibitions per Model:** 5.10
- **Average Obligations per Model:** 5.10
- **Average Policy Size:** 4.93 KB

## Consistency Checking Results

- **Average Intra-Fragment Conflicts:** 0.20
- **Average Inter-Fragment Conflicts:** 0.00

## Policy Reconstruction Results

- **Average Reconstruction Accuracy:** 1.00

## Key Findings

1. The gateway fragmentation strategy produced an average of 6.80 fragments per model.
2. Policy generation took an average of 0.00 seconds per model.
3. The system detected an average of 0.20 conflicts per model.
4. Policy reconstruction achieved an average accuracy of 1.00.

## Conclusion

The BPFragmentODRL system successfully demonstrates the feasibility of fragmenting business processes and generating fragment-level policies using ODRL. The high reconstruction accuracy indicates that the fragment policies effectively capture the original business process policies.

## Visualizations

Visualizations of the evaluation results can be found in the `visualizations` directory.
