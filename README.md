Alex Net (if we have for LeNet, then preferred)

Step 1: Train only for 10 epochs, and save pkl at each and every epoch
(epoch1.pkl, epoch2.pkl, ... , epoch10.pkl)

Step 2: Apply this flatness code on every epoch[i].pkl.

Step 3: Plot them (Expteced result is increse in the flatness as epochs increase)

Homework: Read a bit about the idea of flatness of loss / error surface of deep neural networks



Part 2:  (Lottery ticket hypothesis)

Step 1: Get already pruned weights from LTH (See if you can get in pkl format). Final stage !!
Step 2: Check for compatibility (Make it compatible)
Step 3: Verify the flatness of the lottery ticket


```
Network—init save
|
|__Train---Save Weights---Flatness
|
|__Reinit---Prune—Train---Save Winning Ticket Weight---Flatness
		|
		|__Reinit---Prune—Train---Save Winning Ticket Weight---Flatness
				|
				|__Reinit---Prune—Train---Save Winning Ticket Weight---Flatness
						
```

- No Early Stopping Implemented
- Winning Ticket chosen based on Accuracy



--> Implement sign based reinit
--> Gooey parser