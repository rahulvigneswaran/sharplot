


# (Lottery ticket hypothesis)

Step 1: Get already pruned weights from LTH (See if you can get in pkl format). Final stage !!
Step 2: Check for compatibility (Make it compatible)
Step 3: Verify the flatness of the lottery ticket


```
Network—init save
|
|__Train---Save Weights
|
|__Reinit---Prune—Train---Save Winning Ticket Weight
		|
		|__Reinit---Prune—Train---Save Winning Ticket Weight
				|
				|__Reinit---Prune—Train---Save Winning Ticket Weight
						
```

- No Early Stopping Implemented
- Winning Ticket chosen based on Accuracy



[ ] Implement sign based reinit
[ ] Gooey parser