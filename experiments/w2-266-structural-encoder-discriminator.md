# W2-266 structural encoder discriminator

Status: **REDESIGN**

Decision: **`KILL_REDESIGN structural_capacity`**

## Claim boundary

This experiment may nominate one static encoder candidate for W2-213. It does not establish semantic recombination, dynamic binding, card transfer, executable rules parity, gameplay strength, or integration readiness, and it does not start W2-213.

The preserved static suite contains no runtime objects or legal offers. A nomination
only makes one encoder eligible for W2-213; this run does not start W2-213 or
authorize gameplay integration.

## Sequential path

- Optimization arm fit every seed: **False**.
- Conditional message arm executed: **True**.
- Surviving diagnosis: `structural_capacity`; no encoder is nominated.

## Seed and family receipts

| Arm | Seed | Split | Family | Accuracy | Brier | NLL | ECE-5 |
|---|---:|---|---|---:|---:|---:|---:|
| `bag_v1` | 21401 | train | `order` | 50.0% | 0.2500 | 0.6932 | 0.0041 |
| `bag_v1` | 21401 | train | `hierarchy` | 50.0% | 0.2500 | 0.6932 | 0.0018 |
| `bag_v1` | 21401 | train | `field_role` | 50.0% | 0.2500 | 0.6932 | 0.0014 |
| `bag_v1` | 21401 | train | `argument_binding` | 50.0% | 0.2500 | 0.6932 | 0.0016 |
| `bag_v1` | 21401 | train | `target_choice_role` | 50.0% | 0.2500 | 0.6932 | 0.0017 |
| `bag_v1` | 21401 | validation | `order` | 50.0% | 0.2500 | 0.6932 | 0.0041 |
| `bag_v1` | 21401 | validation | `hierarchy` | 50.0% | 0.2500 | 0.6932 | 0.0017 |
| `bag_v1` | 21401 | validation | `field_role` | 50.0% | 0.2500 | 0.6932 | 0.0015 |
| `bag_v1` | 21401 | validation | `argument_binding` | 50.0% | 0.2500 | 0.6932 | 0.0015 |
| `bag_v1` | 21401 | validation | `target_choice_role` | 50.0% | 0.2500 | 0.6932 | 0.0017 |
| `bag_v1` | 21401 | test | `order` | 50.0% | 0.2500 | 0.6932 | 0.0042 |
| `bag_v1` | 21401 | test | `hierarchy` | 50.0% | 0.2500 | 0.6932 | 0.0019 |
| `bag_v1` | 21401 | test | `field_role` | 50.0% | 0.2500 | 0.6932 | 0.0015 |
| `bag_v1` | 21401 | test | `argument_binding` | 50.0% | 0.2500 | 0.6932 | 0.0016 |
| `bag_v1` | 21401 | test | `target_choice_role` | 50.0% | 0.2500 | 0.6932 | 0.0017 |
| `bag_v1` | 21402 | train | `order` | 50.0% | 0.2500 | 0.6931 | 0.0001 |
| `bag_v1` | 21402 | train | `hierarchy` | 50.0% | 0.2500 | 0.6932 | 0.0022 |
| `bag_v1` | 21402 | train | `field_role` | 50.0% | 0.2500 | 0.6932 | 0.0015 |
| `bag_v1` | 21402 | train | `argument_binding` | 50.0% | 0.2500 | 0.6931 | 0.0006 |
| `bag_v1` | 21402 | train | `target_choice_role` | 50.0% | 0.2500 | 0.6931 | 0.0012 |
| `bag_v1` | 21402 | validation | `order` | 50.0% | 0.2500 | 0.6931 | 0.0002 |
| `bag_v1` | 21402 | validation | `hierarchy` | 50.0% | 0.2500 | 0.6932 | 0.0022 |
| `bag_v1` | 21402 | validation | `field_role` | 50.0% | 0.2500 | 0.6932 | 0.0015 |
| `bag_v1` | 21402 | validation | `argument_binding` | 50.0% | 0.2500 | 0.6931 | 0.0006 |
| `bag_v1` | 21402 | validation | `target_choice_role` | 50.0% | 0.2500 | 0.6931 | 0.0012 |
| `bag_v1` | 21402 | test | `order` | 50.0% | 0.2500 | 0.6931 | 0.0001 |
| `bag_v1` | 21402 | test | `hierarchy` | 50.0% | 0.2500 | 0.6932 | 0.0022 |
| `bag_v1` | 21402 | test | `field_role` | 50.0% | 0.2500 | 0.6932 | 0.0015 |
| `bag_v1` | 21402 | test | `argument_binding` | 50.0% | 0.2500 | 0.6931 | 0.0006 |
| `bag_v1` | 21402 | test | `target_choice_role` | 50.0% | 0.2500 | 0.6931 | 0.0011 |
| `bag_v1` | 21403 | train | `order` | 50.0% | 0.2501 | 0.6934 | 0.0116 |
| `bag_v1` | 21403 | train | `hierarchy` | 50.0% | 0.2500 | 0.6932 | 0.0032 |
| `bag_v1` | 21403 | train | `field_role` | 50.0% | 0.2500 | 0.6932 | 0.0031 |
| `bag_v1` | 21403 | train | `argument_binding` | 50.0% | 0.2500 | 0.6932 | 0.0048 |
| `bag_v1` | 21403 | train | `target_choice_role` | 50.0% | 0.2500 | 0.6932 | 0.0031 |
| `bag_v1` | 21403 | validation | `order` | 50.0% | 0.2501 | 0.6934 | 0.0116 |
| `bag_v1` | 21403 | validation | `hierarchy` | 50.0% | 0.2500 | 0.6932 | 0.0032 |
| `bag_v1` | 21403 | validation | `field_role` | 50.0% | 0.2500 | 0.6932 | 0.0032 |
| `bag_v1` | 21403 | validation | `argument_binding` | 50.0% | 0.2500 | 0.6932 | 0.0048 |
| `bag_v1` | 21403 | validation | `target_choice_role` | 50.0% | 0.2500 | 0.6932 | 0.0031 |
| `bag_v1` | 21403 | test | `order` | 50.0% | 0.2501 | 0.6934 | 0.0116 |
| `bag_v1` | 21403 | test | `hierarchy` | 50.0% | 0.2500 | 0.6932 | 0.0032 |
| `bag_v1` | 21403 | test | `field_role` | 50.0% | 0.2500 | 0.6932 | 0.0032 |
| `bag_v1` | 21403 | test | `argument_binding` | 50.0% | 0.2500 | 0.6932 | 0.0048 |
| `bag_v1` | 21403 | test | `target_choice_role` | 50.0% | 0.2500 | 0.6932 | 0.0031 |
| `bag_v1` | 21404 | train | `order` | 50.0% | 0.2500 | 0.6932 | 0.0024 |
| `bag_v1` | 21404 | train | `hierarchy` | 50.0% | 0.2500 | 0.6932 | 0.0032 |
| `bag_v1` | 21404 | train | `field_role` | 50.0% | 0.2500 | 0.6932 | 0.0027 |
| `bag_v1` | 21404 | train | `argument_binding` | 50.0% | 0.2500 | 0.6932 | 0.0037 |
| `bag_v1` | 21404 | train | `target_choice_role` | 50.0% | 0.2501 | 0.6933 | 0.0075 |
| `bag_v1` | 21404 | validation | `order` | 50.0% | 0.2500 | 0.6932 | 0.0024 |
| `bag_v1` | 21404 | validation | `hierarchy` | 50.0% | 0.2500 | 0.6932 | 0.0033 |
| `bag_v1` | 21404 | validation | `field_role` | 50.0% | 0.2500 | 0.6932 | 0.0026 |
| `bag_v1` | 21404 | validation | `argument_binding` | 50.0% | 0.2500 | 0.6932 | 0.0036 |
| `bag_v1` | 21404 | validation | `target_choice_role` | 50.0% | 0.2501 | 0.6933 | 0.0076 |
| `bag_v1` | 21404 | test | `order` | 50.0% | 0.2500 | 0.6932 | 0.0025 |
| `bag_v1` | 21404 | test | `hierarchy` | 50.0% | 0.2500 | 0.6932 | 0.0032 |
| `bag_v1` | 21404 | test | `field_role` | 50.0% | 0.2500 | 0.6932 | 0.0028 |
| `bag_v1` | 21404 | test | `argument_binding` | 50.0% | 0.2500 | 0.6932 | 0.0037 |
| `bag_v1` | 21404 | test | `target_choice_role` | 50.0% | 0.2501 | 0.6933 | 0.0077 |
| `bag_v1` | 21405 | train | `order` | 50.0% | 0.2500 | 0.6931 | 0.0010 |
| `bag_v1` | 21405 | train | `hierarchy` | 50.0% | 0.2500 | 0.6931 | 0.0003 |
| `bag_v1` | 21405 | train | `field_role` | 50.0% | 0.2500 | 0.6932 | 0.0012 |
| `bag_v1` | 21405 | train | `argument_binding` | 50.0% | 0.2500 | 0.6932 | 0.0014 |
| `bag_v1` | 21405 | train | `target_choice_role` | 50.0% | 0.2500 | 0.6932 | 0.0017 |
| `bag_v1` | 21405 | validation | `order` | 50.0% | 0.2500 | 0.6931 | 0.0011 |
| `bag_v1` | 21405 | validation | `hierarchy` | 50.0% | 0.2500 | 0.6931 | 0.0003 |
| `bag_v1` | 21405 | validation | `field_role` | 50.0% | 0.2500 | 0.6931 | 0.0011 |
| `bag_v1` | 21405 | validation | `argument_binding` | 50.0% | 0.2500 | 0.6932 | 0.0014 |
| `bag_v1` | 21405 | validation | `target_choice_role` | 50.0% | 0.2500 | 0.6932 | 0.0016 |
| `bag_v1` | 21405 | test | `order` | 50.0% | 0.2500 | 0.6931 | 0.0010 |
| `bag_v1` | 21405 | test | `hierarchy` | 50.0% | 0.2500 | 0.6931 | 0.0004 |
| `bag_v1` | 21405 | test | `field_role` | 50.0% | 0.2500 | 0.6932 | 0.0013 |
| `bag_v1` | 21405 | test | `argument_binding` | 50.0% | 0.2500 | 0.6932 | 0.0014 |
| `bag_v1` | 21405 | test | `target_choice_role` | 50.0% | 0.2500 | 0.6932 | 0.0018 |
| `relational_message_encoder_v1` | 21401 | train | `order` | 100.0% | 0.0000 | 0.0001 | 0.0001 |
| `relational_message_encoder_v1` | 21401 | train | `hierarchy` | 100.0% | 0.0000 | 0.0001 | 0.0001 |
| `relational_message_encoder_v1` | 21401 | train | `field_role` | 50.0% | 0.2500 | 0.6932 | 0.0020 |
| `relational_message_encoder_v1` | 21401 | train | `argument_binding` | 100.0% | 0.0000 | 0.0002 | 0.0002 |
| `relational_message_encoder_v1` | 21401 | train | `target_choice_role` | 100.0% | 0.0000 | 0.0002 | 0.0002 |
| `relational_message_encoder_v1` | 21401 | validation | `order` | 100.0% | 0.0000 | 0.0001 | 0.0001 |
| `relational_message_encoder_v1` | 21401 | validation | `hierarchy` | 100.0% | 0.0000 | 0.0001 | 0.0001 |
| `relational_message_encoder_v1` | 21401 | validation | `field_role` | 50.0% | 0.2500 | 0.6932 | 0.0020 |
| `relational_message_encoder_v1` | 21401 | validation | `argument_binding` | 100.0% | 0.0000 | 0.0002 | 0.0002 |
| `relational_message_encoder_v1` | 21401 | validation | `target_choice_role` | 100.0% | 0.0000 | 0.0002 | 0.0002 |
| `relational_message_encoder_v1` | 21401 | test | `order` | 100.0% | 0.0000 | 0.0001 | 0.0001 |
| `relational_message_encoder_v1` | 21401 | test | `hierarchy` | 100.0% | 0.0000 | 0.0001 | 0.0001 |
| `relational_message_encoder_v1` | 21401 | test | `field_role` | 50.0% | 0.2500 | 0.6932 | 0.0020 |
| `relational_message_encoder_v1` | 21401 | test | `argument_binding` | 100.0% | 0.0000 | 0.0002 | 0.0002 |
| `relational_message_encoder_v1` | 21401 | test | `target_choice_role` | 100.0% | 0.0000 | 0.0002 | 0.0002 |
| `relational_message_encoder_v1` | 21402 | train | `order` | 100.0% | 0.0000 | 0.0001 | 0.0001 |
| `relational_message_encoder_v1` | 21402 | train | `hierarchy` | 100.0% | 0.0000 | 0.0001 | 0.0001 |
| `relational_message_encoder_v1` | 21402 | train | `field_role` | 50.0% | 0.2500 | 0.6932 | 0.0032 |
| `relational_message_encoder_v1` | 21402 | train | `argument_binding` | 100.0% | 0.0000 | 0.0002 | 0.0002 |
| `relational_message_encoder_v1` | 21402 | train | `target_choice_role` | 100.0% | 0.0000 | 0.0002 | 0.0002 |
| `relational_message_encoder_v1` | 21402 | validation | `order` | 100.0% | 0.0000 | 0.0001 | 0.0001 |
| `relational_message_encoder_v1` | 21402 | validation | `hierarchy` | 100.0% | 0.0000 | 0.0001 | 0.0001 |
| `relational_message_encoder_v1` | 21402 | validation | `field_role` | 50.0% | 0.2500 | 0.6932 | 0.0032 |
| `relational_message_encoder_v1` | 21402 | validation | `argument_binding` | 100.0% | 0.0000 | 0.0002 | 0.0002 |
| `relational_message_encoder_v1` | 21402 | validation | `target_choice_role` | 100.0% | 0.0000 | 0.0002 | 0.0002 |
| `relational_message_encoder_v1` | 21402 | test | `order` | 100.0% | 0.0000 | 0.0001 | 0.0001 |
| `relational_message_encoder_v1` | 21402 | test | `hierarchy` | 100.0% | 0.0000 | 0.0001 | 0.0001 |
| `relational_message_encoder_v1` | 21402 | test | `field_role` | 50.0% | 0.2500 | 0.6932 | 0.0032 |
| `relational_message_encoder_v1` | 21402 | test | `argument_binding` | 100.0% | 0.0000 | 0.0002 | 0.0002 |
| `relational_message_encoder_v1` | 21402 | test | `target_choice_role` | 100.0% | 0.0000 | 0.0002 | 0.0002 |
| `relational_message_encoder_v1` | 21403 | train | `order` | 100.0% | 0.0000 | 0.0001 | 0.0001 |
| `relational_message_encoder_v1` | 21403 | train | `hierarchy` | 100.0% | 0.0000 | 0.0001 | 0.0001 |
| `relational_message_encoder_v1` | 21403 | train | `field_role` | 74.0% | 0.1509 | 0.4217 | 0.0243 |
| `relational_message_encoder_v1` | 21403 | train | `argument_binding` | 100.0% | 0.0000 | 0.0006 | 0.0006 |
| `relational_message_encoder_v1` | 21403 | train | `target_choice_role` | 100.0% | 0.0000 | 0.0004 | 0.0004 |
| `relational_message_encoder_v1` | 21403 | validation | `order` | 100.0% | 0.0000 | 0.0001 | 0.0001 |
| `relational_message_encoder_v1` | 21403 | validation | `hierarchy` | 100.0% | 0.0000 | 0.0001 | 0.0001 |
| `relational_message_encoder_v1` | 21403 | validation | `field_role` | 59.4% | 0.2176 | 0.6047 | 0.0028 |
| `relational_message_encoder_v1` | 21403 | validation | `argument_binding` | 100.0% | 0.0000 | 0.0006 | 0.0006 |
| `relational_message_encoder_v1` | 21403 | validation | `target_choice_role` | 100.0% | 0.0000 | 0.0004 | 0.0004 |
| `relational_message_encoder_v1` | 21403 | test | `order` | 100.0% | 0.0000 | 0.0001 | 0.0001 |
| `relational_message_encoder_v1` | 21403 | test | `hierarchy` | 100.0% | 0.0000 | 0.0001 | 0.0001 |
| `relational_message_encoder_v1` | 21403 | test | `field_role` | 71.9% | 0.1548 | 0.4314 | 0.0083 |
| `relational_message_encoder_v1` | 21403 | test | `argument_binding` | 100.0% | 0.0000 | 0.0006 | 0.0006 |
| `relational_message_encoder_v1` | 21403 | test | `target_choice_role` | 100.0% | 0.0000 | 0.0004 | 0.0004 |
| `relational_message_encoder_v1` | 21404 | train | `order` | 100.0% | 0.0000 | 0.0002 | 0.0002 |
| `relational_message_encoder_v1` | 21404 | train | `hierarchy` | 100.0% | 0.0000 | 0.0001 | 0.0001 |
| `relational_message_encoder_v1` | 21404 | train | `field_role` | 100.0% | 0.0000 | 0.0004 | 0.0004 |
| `relational_message_encoder_v1` | 21404 | train | `argument_binding` | 100.0% | 0.0000 | 0.0002 | 0.0002 |
| `relational_message_encoder_v1` | 21404 | train | `target_choice_role` | 100.0% | 0.0000 | 0.0002 | 0.0002 |
| `relational_message_encoder_v1` | 21404 | validation | `order` | 100.0% | 0.0000 | 0.0002 | 0.0002 |
| `relational_message_encoder_v1` | 21404 | validation | `hierarchy` | 100.0% | 0.0000 | 0.0001 | 0.0001 |
| `relational_message_encoder_v1` | 21404 | validation | `field_role` | 100.0% | 0.0000 | 0.0004 | 0.0004 |
| `relational_message_encoder_v1` | 21404 | validation | `argument_binding` | 100.0% | 0.0000 | 0.0002 | 0.0002 |
| `relational_message_encoder_v1` | 21404 | validation | `target_choice_role` | 100.0% | 0.0000 | 0.0002 | 0.0002 |
| `relational_message_encoder_v1` | 21404 | test | `order` | 100.0% | 0.0000 | 0.0002 | 0.0002 |
| `relational_message_encoder_v1` | 21404 | test | `hierarchy` | 100.0% | 0.0000 | 0.0001 | 0.0001 |
| `relational_message_encoder_v1` | 21404 | test | `field_role` | 100.0% | 0.0000 | 0.0004 | 0.0004 |
| `relational_message_encoder_v1` | 21404 | test | `argument_binding` | 100.0% | 0.0000 | 0.0002 | 0.0002 |
| `relational_message_encoder_v1` | 21404 | test | `target_choice_role` | 96.9% | 0.0305 | 0.1386 | 0.0307 |
| `relational_message_encoder_v1` | 21405 | train | `order` | 100.0% | 0.0000 | 0.0007 | 0.0007 |
| `relational_message_encoder_v1` | 21405 | train | `hierarchy` | 100.0% | 0.0000 | 0.0013 | 0.0013 |
| `relational_message_encoder_v1` | 21405 | train | `field_role` | 100.0% | 0.0000 | 0.0024 | 0.0024 |
| `relational_message_encoder_v1` | 21405 | train | `argument_binding` | 100.0% | 0.0000 | 0.0049 | 0.0048 |
| `relational_message_encoder_v1` | 21405 | train | `target_choice_role` | 77.1% | 0.1161 | 0.3443 | 0.0373 |
| `relational_message_encoder_v1` | 21405 | validation | `order` | 100.0% | 0.0000 | 0.0007 | 0.0007 |
| `relational_message_encoder_v1` | 21405 | validation | `hierarchy` | 100.0% | 0.0000 | 0.0013 | 0.0013 |
| `relational_message_encoder_v1` | 21405 | validation | `field_role` | 96.9% | 0.0311 | 0.1876 | 0.0288 |
| `relational_message_encoder_v1` | 21405 | validation | `argument_binding` | 100.0% | 0.0000 | 0.0049 | 0.0048 |
| `relational_message_encoder_v1` | 21405 | validation | `target_choice_role` | 71.9% | 0.1420 | 0.4117 | 0.0351 |
| `relational_message_encoder_v1` | 21405 | test | `order` | 100.0% | 0.0000 | 0.0007 | 0.0007 |
| `relational_message_encoder_v1` | 21405 | test | `hierarchy` | 100.0% | 0.0000 | 0.0013 | 0.0013 |
| `relational_message_encoder_v1` | 21405 | test | `field_role` | 100.0% | 0.0000 | 0.0024 | 0.0024 |
| `relational_message_encoder_v1` | 21405 | test | `argument_binding` | 100.0% | 0.0000 | 0.0049 | 0.0049 |
| `relational_message_encoder_v1` | 21405 | test | `target_choice_role` | 71.9% | 0.1420 | 0.4117 | 0.0351 |
| `relational_semantic_encoder_v1_opt4000` | 21401 | train | `order` | 100.0% | 0.0000 | 0.0002 | 0.0002 |
| `relational_semantic_encoder_v1_opt4000` | 21401 | train | `hierarchy` | 100.0% | 0.0000 | 0.0001 | 0.0001 |
| `relational_semantic_encoder_v1_opt4000` | 21401 | train | `field_role` | 76.0% | 0.1511 | 0.4277 | 0.0161 |
| `relational_semantic_encoder_v1_opt4000` | 21401 | train | `argument_binding` | 50.0% | 0.2500 | 0.6932 | 0.0034 |
| `relational_semantic_encoder_v1_opt4000` | 21401 | train | `target_choice_role` | 50.0% | 0.2500 | 0.6932 | 0.0058 |
| `relational_semantic_encoder_v1_opt4000` | 21401 | validation | `order` | 100.0% | 0.0000 | 0.0002 | 0.0002 |
| `relational_semantic_encoder_v1_opt4000` | 21401 | validation | `hierarchy` | 100.0% | 0.0000 | 0.0001 | 0.0001 |
| `relational_semantic_encoder_v1_opt4000` | 21401 | validation | `field_role` | 71.9% | 0.1518 | 0.4196 | 0.0484 |
| `relational_semantic_encoder_v1_opt4000` | 21401 | validation | `argument_binding` | 50.0% | 0.2500 | 0.6932 | 0.0034 |
| `relational_semantic_encoder_v1_opt4000` | 21401 | validation | `target_choice_role` | 50.0% | 0.2500 | 0.6932 | 0.0058 |
| `relational_semantic_encoder_v1_opt4000` | 21401 | test | `order` | 100.0% | 0.0000 | 0.0002 | 0.0002 |
| `relational_semantic_encoder_v1_opt4000` | 21401 | test | `hierarchy` | 100.0% | 0.0000 | 0.0001 | 0.0001 |
| `relational_semantic_encoder_v1_opt4000` | 21401 | test | `field_role` | 75.0% | 0.1500 | 0.4218 | 0.0047 |
| `relational_semantic_encoder_v1_opt4000` | 21401 | test | `argument_binding` | 50.0% | 0.2500 | 0.6932 | 0.0034 |
| `relational_semantic_encoder_v1_opt4000` | 21401 | test | `target_choice_role` | 50.0% | 0.2500 | 0.6932 | 0.0058 |
| `relational_semantic_encoder_v1_opt4000` | 21402 | train | `order` | 100.0% | 0.0000 | 0.0001 | 0.0001 |
| `relational_semantic_encoder_v1_opt4000` | 21402 | train | `hierarchy` | 100.0% | 0.0000 | 0.0001 | 0.0001 |
| `relational_semantic_encoder_v1_opt4000` | 21402 | train | `field_role` | 50.0% | 0.2500 | 0.6932 | 0.0033 |
| `relational_semantic_encoder_v1_opt4000` | 21402 | train | `argument_binding` | 100.0% | 0.0000 | 0.0002 | 0.0002 |
| `relational_semantic_encoder_v1_opt4000` | 21402 | train | `target_choice_role` | 100.0% | 0.0000 | 0.0002 | 0.0002 |
| `relational_semantic_encoder_v1_opt4000` | 21402 | validation | `order` | 100.0% | 0.0000 | 0.0001 | 0.0001 |
| `relational_semantic_encoder_v1_opt4000` | 21402 | validation | `hierarchy` | 100.0% | 0.0000 | 0.0001 | 0.0001 |
| `relational_semantic_encoder_v1_opt4000` | 21402 | validation | `field_role` | 50.0% | 0.2500 | 0.6932 | 0.0033 |
| `relational_semantic_encoder_v1_opt4000` | 21402 | validation | `argument_binding` | 100.0% | 0.0000 | 0.0002 | 0.0002 |
| `relational_semantic_encoder_v1_opt4000` | 21402 | validation | `target_choice_role` | 100.0% | 0.0000 | 0.0002 | 0.0002 |
| `relational_semantic_encoder_v1_opt4000` | 21402 | test | `order` | 100.0% | 0.0000 | 0.0001 | 0.0001 |
| `relational_semantic_encoder_v1_opt4000` | 21402 | test | `hierarchy` | 100.0% | 0.0000 | 0.0001 | 0.0001 |
| `relational_semantic_encoder_v1_opt4000` | 21402 | test | `field_role` | 50.0% | 0.2500 | 0.6932 | 0.0033 |
| `relational_semantic_encoder_v1_opt4000` | 21402 | test | `argument_binding` | 100.0% | 0.0000 | 0.0002 | 0.0002 |
| `relational_semantic_encoder_v1_opt4000` | 21402 | test | `target_choice_role` | 100.0% | 0.0000 | 0.0002 | 0.0002 |
| `relational_semantic_encoder_v1_opt4000` | 21403 | train | `order` | 100.0% | 0.0000 | 0.0002 | 0.0002 |
| `relational_semantic_encoder_v1_opt4000` | 21403 | train | `hierarchy` | 100.0% | 0.0000 | 0.0001 | 0.0001 |
| `relational_semantic_encoder_v1_opt4000` | 21403 | train | `field_role` | 50.0% | 0.2500 | 0.6932 | 0.0050 |
| `relational_semantic_encoder_v1_opt4000` | 21403 | train | `argument_binding` | 50.0% | 0.2500 | 0.6932 | 0.0063 |
| `relational_semantic_encoder_v1_opt4000` | 21403 | train | `target_choice_role` | 50.0% | 0.2500 | 0.6931 | 0.0001 |
| `relational_semantic_encoder_v1_opt4000` | 21403 | validation | `order` | 100.0% | 0.0000 | 0.0002 | 0.0002 |
| `relational_semantic_encoder_v1_opt4000` | 21403 | validation | `hierarchy` | 100.0% | 0.0000 | 0.0001 | 0.0001 |
| `relational_semantic_encoder_v1_opt4000` | 21403 | validation | `field_role` | 50.0% | 0.2500 | 0.6932 | 0.0050 |
| `relational_semantic_encoder_v1_opt4000` | 21403 | validation | `argument_binding` | 50.0% | 0.2500 | 0.6932 | 0.0063 |
| `relational_semantic_encoder_v1_opt4000` | 21403 | validation | `target_choice_role` | 50.0% | 0.2500 | 0.6931 | 0.0001 |
| `relational_semantic_encoder_v1_opt4000` | 21403 | test | `order` | 100.0% | 0.0000 | 0.0002 | 0.0002 |
| `relational_semantic_encoder_v1_opt4000` | 21403 | test | `hierarchy` | 100.0% | 0.0000 | 0.0001 | 0.0001 |
| `relational_semantic_encoder_v1_opt4000` | 21403 | test | `field_role` | 50.0% | 0.2500 | 0.6932 | 0.0050 |
| `relational_semantic_encoder_v1_opt4000` | 21403 | test | `argument_binding` | 50.0% | 0.2500 | 0.6932 | 0.0063 |
| `relational_semantic_encoder_v1_opt4000` | 21403 | test | `target_choice_role` | 50.0% | 0.2500 | 0.6931 | 0.0001 |
| `relational_semantic_encoder_v1_opt4000` | 21404 | train | `order` | 100.0% | 0.0000 | 0.0001 | 0.0001 |
| `relational_semantic_encoder_v1_opt4000` | 21404 | train | `hierarchy` | 100.0% | 0.0000 | 0.0002 | 0.0002 |
| `relational_semantic_encoder_v1_opt4000` | 21404 | train | `field_role` | 70.8% | 0.1465 | 0.4199 | 0.0230 |
| `relational_semantic_encoder_v1_opt4000` | 21404 | train | `argument_binding` | 100.0% | 0.0000 | 0.0007 | 0.0007 |
| `relational_semantic_encoder_v1_opt4000` | 21404 | train | `target_choice_role` | 100.0% | 0.0000 | 0.0008 | 0.0008 |
| `relational_semantic_encoder_v1_opt4000` | 21404 | validation | `order` | 100.0% | 0.0000 | 0.0001 | 0.0001 |
| `relational_semantic_encoder_v1_opt4000` | 21404 | validation | `hierarchy` | 100.0% | 0.0000 | 0.0002 | 0.0002 |
| `relational_semantic_encoder_v1_opt4000` | 21404 | validation | `field_role` | 56.2% | 0.2191 | 0.6115 | 0.0166 |
| `relational_semantic_encoder_v1_opt4000` | 21404 | validation | `argument_binding` | 100.0% | 0.0000 | 0.0007 | 0.0007 |
| `relational_semantic_encoder_v1_opt4000` | 21404 | validation | `target_choice_role` | 100.0% | 0.0000 | 0.0008 | 0.0008 |
| `relational_semantic_encoder_v1_opt4000` | 21404 | test | `order` | 100.0% | 0.0000 | 0.0001 | 0.0001 |
| `relational_semantic_encoder_v1_opt4000` | 21404 | test | `hierarchy` | 100.0% | 0.0000 | 0.0003 | 0.0003 |
| `relational_semantic_encoder_v1_opt4000` | 21404 | test | `field_role` | 62.5% | 0.1880 | 0.5294 | 0.0194 |
| `relational_semantic_encoder_v1_opt4000` | 21404 | test | `argument_binding` | 100.0% | 0.0000 | 0.0007 | 0.0007 |
| `relational_semantic_encoder_v1_opt4000` | 21404 | test | `target_choice_role` | 100.0% | 0.0000 | 0.0008 | 0.0008 |
| `relational_semantic_encoder_v1_opt4000` | 21405 | train | `order` | 100.0% | 0.0000 | 0.0005 | 0.0005 |
| `relational_semantic_encoder_v1_opt4000` | 21405 | train | `hierarchy` | 100.0% | 0.0000 | 0.0006 | 0.0006 |
| `relational_semantic_encoder_v1_opt4000` | 21405 | train | `field_role` | 64.6% | 0.2077 | 0.5847 | 0.0186 |
| `relational_semantic_encoder_v1_opt4000` | 21405 | train | `argument_binding` | 100.0% | 0.0000 | 0.0024 | 0.0024 |
| `relational_semantic_encoder_v1_opt4000` | 21405 | train | `target_choice_role` | 100.0% | 0.0000 | 0.0032 | 0.0032 |
| `relational_semantic_encoder_v1_opt4000` | 21405 | validation | `order` | 100.0% | 0.0000 | 0.0005 | 0.0005 |
| `relational_semantic_encoder_v1_opt4000` | 21405 | validation | `hierarchy` | 100.0% | 0.0000 | 0.0006 | 0.0006 |
| `relational_semantic_encoder_v1_opt4000` | 21405 | validation | `field_role` | 56.2% | 0.2344 | 0.6519 | 0.0355 |
| `relational_semantic_encoder_v1_opt4000` | 21405 | validation | `argument_binding` | 100.0% | 0.0000 | 0.0024 | 0.0024 |
| `relational_semantic_encoder_v1_opt4000` | 21405 | validation | `target_choice_role` | 100.0% | 0.0000 | 0.0032 | 0.0032 |
| `relational_semantic_encoder_v1_opt4000` | 21405 | test | `order` | 100.0% | 0.0000 | 0.0005 | 0.0005 |
| `relational_semantic_encoder_v1_opt4000` | 21405 | test | `hierarchy` | 100.0% | 0.0000 | 0.0006 | 0.0006 |
| `relational_semantic_encoder_v1_opt4000` | 21405 | test | `field_role` | 62.5% | 0.2143 | 0.6014 | 0.0062 |
| `relational_semantic_encoder_v1_opt4000` | 21405 | test | `argument_binding` | 100.0% | 0.0000 | 0.0024 | 0.0024 |
| `relational_semantic_encoder_v1_opt4000` | 21405 | test | `target_choice_role` | 100.0% | 0.0000 | 0.0032 | 0.0032 |

## Training and CPU receipts

| Arm | Seed | First 99% step | Max train | Selected step | Parameters | model p50/p95 µs | online p50/p95 µs | cached p50/p95 µs | cached batch-128/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bag_v1` | 21401 | None | 50.0% | 540 | 9128 | 85.3/359.7 | 148.1/1019.2 | 125.6/1147.9 | 55743 |
| `bag_v1` | 21402 | None | 50.0% | 520 | 9128 | 86.3/475.1 | 142.2/442.6 | 121.5/556.6 | 60409 |
| `bag_v1` | 21403 | None | 50.0% | 540 | 9128 | 87.1/366.7 | 141.0/455.3 | 121.9/539.9 | 75899 |
| `bag_v1` | 21404 | None | 50.0% | 540 | 9128 | 85.8/315.9 | 144.5/622.7 | 123.2/495.8 | 75311 |
| `bag_v1` | 21405 | None | 50.0% | 620 | 9128 | 87.1/423.7 | 142.9/683.2 | 121.3/186.7 | 143776 |
| `relational_message_encoder_v1` | 21401 | None | 90.0% | 3960 | 9030 | 192.2/214.8 | 525.4/586.8 | 210.3/229.5 | 10907 |
| `relational_message_encoder_v1` | 21402 | None | 90.0% | 3980 | 9030 | 190.2/205.7 | 528.0/595.9 | 213.2/269.8 | 10610 |
| `relational_message_encoder_v1` | 21403 | None | 94.8% | 3940 | 9030 | 193.4/232.9 | 527.6/614.8 | 209.9/262.0 | 10505 |
| `relational_message_encoder_v1` | 21404 | 560 | 100.0% | 4000 | 9030 | 192.1/216.6 | 533.6/680.3 | 209.5/239.0 | 10855 |
| `relational_message_encoder_v1` | 21405 | None | 95.4% | 1440 | 9030 | 190.6/205.4 | 518.9/538.6 | 209.4/240.9 | 10552 |
| `relational_semantic_encoder_v1_opt4000` | 21401 | None | 75.2% | 3880 | 8838 | 189.3/246.2 | 492.7/606.3 | 174.9/214.1 | 13416 |
| `relational_semantic_encoder_v1_opt4000` | 21402 | None | 90.0% | 3980 | 8838 | 157.7/165.3 | 493.2/648.1 | 175.9/239.0 | 13621 |
| `relational_semantic_encoder_v1_opt4000` | 21403 | None | 71.5% | 3780 | 8838 | 155.6/174.2 | 496.5/658.1 | 174.6/214.8 | 13341 |
| `relational_semantic_encoder_v1_opt4000` | 21404 | None | 94.2% | 4000 | 8838 | 158.4/215.7 | 491.0/562.2 | 174.0/207.9 | 13507 |
| `relational_semantic_encoder_v1_opt4000` | 21405 | None | 92.9% | 1780 | 8838 | 153.8/165.9 | 489.3/522.5 | 173.2/187.8 | 14139 |

## Gates and diagnosis

- Bag exact symmetry and 50% ceiling: **True**.
- Selected semantic and calibration admission: **not reached** because neither
  bounded structural arm fit the known training signal in all five seeds.
- Parameter match: **passed** for both structural arms (3.177% optimization,
  1.074% message; 5% maximum).
- Optimization arm cached CPU: p95 **1.006x** (passes 2.5x), throughput
  **0.098x** (fails 0.4x); model-only throughput **0.088x**.
- Message arm cached CPU: p95 **1.290x** (passes 2.5x), throughput **0.073x**
  (fails 0.4x); model-only throughput **0.071x**.
- CPU results are attribution-only after the earlier capacity stop and do not
  change the terminal diagnosis.
- Cold tensor-catalog build: 560.343 ms (not amortized).
- Cached projection matched online tensors, logits, and metrics exactly for all 800 programs and all executed seeds/arms before timing.

## Provenance and budget

- Preregistration revision: `661392e7e09a0db902fea8de3e9392cad24ab6a1`.
- Contract SHA-256: `bb2466d07ccd5b9be19a57d03a0d89b8b8eebc8b1c8ce65a899512567d4c586c`.
- Suite SHA-256: `5595ce579017c4ec84b8746cb30a3f4bb09a69e4801ccaf7748467f7bec2f948`.
- Optimizer steps: 44000 / 44000.
- Presented examples: 2816000 / 2816000.
- Wall clock: 763.16s / 1800s.

## Result-contingent next step

`KILL_REDESIGN structural_capacity`

The named redesign diagnosis is terminal for this bounded experiment; no gate was relaxed and W2-213 remains blocked.
