# API Client Code Generation

## Using Swagger

TODO: provide more context and info on how/why

  ```python
  import swagger_client
  print swagger_client.DefaultApi().resnet18_predict('white-sleeping-kitten.jpg')
  ```
  ```
  {
    'prediction':
      "[[{u'class': u'n02123045 tabby, tabby cat', u'probability': 0.3166358768939972}, {u'class': u'n02124075 Egyptian cat', u'probability': 0.3160117268562317}, {u'class': u'n04074963 remote control, remote', u'probability': 0.047916918992996216}, {u'class': u'n02123159 tiger cat', u'probability': 0.036371976137161255}, {u'class': u'n02127052 lynx, catamount', u'probability': 0.03163142874836922}]]"
  }
  ```
