# Model dfamp
This model predicts the amount of a certain food based on the history.

#### Notes on the implementation
In the beginning I tried a **Regression Model** but that **didn't work very well**. <br/>
It didn't work because Regression calculates **a continuous variable**, which looks ok for this scenario (calculating an amount), but **is not actually what I want out of it**. 

An example: <br/>
I usually eat **200gr of Greek Yogurt**. That happens most of the time, but sometimes it could happen that it's less or more. <br/>
Still, I would like the model **to suggest 200gr** because that's the most likely amount. <br/>

With a regression model, 200 would never be the result, because it estimates a continuous variable. In my first attempts, the best model kept estimating **192gr** with a very high accuracy.

So I ended up using a **Classification Model**: I want the estimate for yogurt to fall in the "200gr category".