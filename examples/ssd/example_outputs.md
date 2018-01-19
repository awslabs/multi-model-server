# SSD Example Outputs

### Dog Beach

![dog beach](https://farm9.staticflickr.com/8184/8081332083_2a9a971149_o_d.jpg)
```bash
curl -o dogbeach.jpg https://farm9.staticflickr.com/8184/8081332083_2a9a971149_o_d.jpg
curl -X POST http://127.0.0.1:8080/ssd/predict -F "data=@dogbeach.jpg"
{
  "prediction": [
    [
      "dog",
      2484,
      1300,
      2996,
      1741
    ],
    [
      "person",
      1513,
      1580,
      1846,
      2584
    ],
    [
      "person",
      810,
      1561,
      1076,
      2161
    ],
    [
      "person",
      1151,
      86,
      1403,
      732
    ],
    [
      "person",
      3934,
      228,
      4169,
      766
    ],
    [
      "dog",
      591,
      416,
      976,
      722
    ],
    [
      "horse",
      3456,
      27,
      3912,
      300
    ],
    [
      "dog",
      2186,
      1381,
      2589,
      1720
    ],
    [
      "person",
      371,
      2771,
      671,
      3168
    ],
    [
      "person",
      522,
      662,
      716,
      1152
    ]
  ]
}
```

### 3 Dogs on Beach
![3 dogs on beach](https://farm9.staticflickr.com/8051/8081326814_991e7b15cc_o_d.jpg)
```bash
curl -o 3dogs.jpg https://farm9.staticflickr.com/8051/8081326814_991e7b15cc_o_d.jpg
curl -X POST http://127.0.0.1:8080/ssd/predict -F "data=@3dogs.jpg"
{
  "prediction": [
    [
      "dog",
      2967,
      953,
      4228,
      2125
    ],
    [
      "dog",
      2058,
      1454,
      3123,
      2130
    ],
    [
      "cow",
      1508,
      881,
      2206,
      2023
    ]
  ]
}
```
### Sailboat
![sailboat](https://farm9.staticflickr.com/8316/7990362092_7e4a1cebb4_o_d.jpg)
```bash
curl -o sailboat.jpg https://farm9.staticflickr.com/8316/7990362092_7e4a1cebb4_o_d.jpg
curl -X POST http://127.0.0.1:8080/ssd/predict -F "data=@sailboat.jpg"
{
  "prediction": [
    [
      "boat",
      1177,
      597,
      1872,
      2346
    ],
    [
      "boat",
      1241,
      1987,
      1833,
      2256
    ]
  ]
}
```
