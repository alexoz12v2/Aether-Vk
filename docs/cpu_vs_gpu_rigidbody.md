# Why We Use the CPU (Not the GPU) for Rigid Body Collisions

If you are wondering why our ultra-fast GPU physics engine suddenly passes the baton back to the CPU just to calculate collisions between large, chunky objects (Rigid Bodies), you are not alone! It sounds counter-intuitive at first. Shouldn't the graphics card do *everything*?

To understand why, we need to understand how CPUs and GPUs think.

## The GPU: The Marching Band

Imagine the GPU as a massive, perfectly synchronized marching band of 10,000 people. 

When the band director shouts "Take one step forward!" all 10,000 people take one step forward at the exact same millisecond. This is amazing for tasks where everyone does the exact same thing—like calculating the gravity for 10 million tiny, identical dust particles. 

However, there is a catch: **they must do the exact same thing at the exact same time.**

If one person in the band drops their hat and needs to bend down to pick it up, the *entire marching band has to stop and wait for them*. The GPU cannot easily handle situations where one worker needs to do something different from the person next to them. In computer science, we call this **Branching** (using `if / else` statements), and when the marching band gets out of sync, we call it **Warp Divergence**. It kills GPU performance.

## The CPU: The Smart Independent Contractors

Imagine the CPU as a small team of 8 to 16 highly skilled, completely independent contractors. 

They don't march in sync. One can be painting a wall, another can be doing plumbing, and another can be eating a sandwich. They are incredibly fast at solving complex, multi-step puzzles that require making a lot of different decisions based on what they find along the way.

## Why Rigid Bodies are a Nightmare for the Marching Band

When two complex shapes (like a spaceship and an asteroid) crash into each other, figuring out exactly *where* they touched and *how hard* they hit is a very complex puzzle. 

To solve it, we use advanced mathematical algorithms with scary names like **GJK** (Gilbert-Johnson-Keerthi) and **Voronoi Clip**. 

Here is how those algorithms work in plain English:
1. Look at a point on Object A.
2. *If* it's near an edge of Object B, do some math.
3. *Else if* it's near a flat face of Object B, do completely different math.
4. *Else*, loop back and try another point.
5. Keep guessing, checking, and looping until you find the exact spot they touch.

This requires hundreds of `if / else` decisions. It is a highly unpredictable maze. 

If we gave this task to the GPU Marching Band, Worker #1 might find the answer in 2 steps. Worker #2 might get stuck in a loop and take 50 steps. But because they are a marching band, Worker #1 (and the rest of the 10,000 workers) **have to freeze and wait** doing absolutely nothing until Worker #2 finishes their 50 steps. 

The GPU would grind to a halt.

## The Perfect Teamwork

Because our simulation usually only has a handful of complex Rigid Bodies (like a few landers or asteroids) but *millions* of simple particles (like dust or thraster exhaust):

1. We give the millions of simple, predictable particles to the **GPU Marching Band**. They blast through the math in milliseconds.
2. We give the few complex, unpredictable spaceship collisions (GJK / Voronoi) to the **CPU Independent Contractors**. They use their smarts to solve the maze instantly without holding anyone else back.

By playing to the strengths of both processors, we get the fastest possible physics engine!
