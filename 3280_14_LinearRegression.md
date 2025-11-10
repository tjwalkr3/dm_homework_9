## Least Squares, Regression, and the pseudo-inverse
$$Ax=b$$

$A$ is known, $b$ is known, and we want to know $x$. We have typically looked at this where $A$ is a square. If it's not square, it's been difficult to calculate.

But the SVD allows us to generalize this for non-square $A$ matrices.

An __underdetermined__ matrix is one where $n<m$ (a short, fat matrix). In this case, there are more variables than there are equations. Not enough measurements in $b$ to determine a single, unique solution $x$. This often (not always) gives an infinite number of solutions.

An __overdetermined__ matrix is one where $n>m$ (a tall, skinny matrix). In this case, there are more equations than needed. In general, there are too many measurements in $b$, so there is no unique solution $x$. Again, not always true.

SVD allows us to "invert" $A$ (known as a pseudo-inverse) which can be used to approximate $x$.

$$A=\hat{U}\hat{\Sigma}V^T$$
$$Ax=b$$
$$\hat{U}\hat{\Sigma}V^Tx=b$$
$$V\Sigma^{-1}U^T U\Sigma V^T x = V\Sigma^{-1}U^T b$$
$$x = V\Sigma^{-1}U^T b$$
$$x = A^\dagger b$$

where $A^\dagger$ is known as the __Moore-Penrose__ (left) __Pseudo Inverse__. We can use $A^\dagger$ to approximate $x$
$$\tilde{x} = A^\dagger b$$

In the underdetermined case, there are an infinite number of solutions. Which one is right? We generally say that the *minimum-norm solution* is the one where $\min{||\tilde{x}||_2}$ such that $A\tilde{x}=b$.

In the overdetermined solution, we can find the $\tilde{x}$ that minimizes the error:
$$\min||A\tilde{x}-b||_2$$
This is known as the least squares solution.

So, how well does this work? To find out, let's plug everything back in:
$$A=\hat{U}\hat{\Sigma}V^T \qquad \tilde{x} = A^\dagger b = V\Sigma^{-1}U^T b$$
$$A\tilde{x} = \hat{U}\Sigma V^T V\Sigma^{-1} \hat{U}^T b$$
$$A\tilde{x} = \hat{U}\hat{U}^T b$$

Remember that $\hat{U}$ is not unitary, so $\hat{U}\hat{U}^T\ne \mathbb{I}$. So what is this? $\hat{U}\hat{U}^T b$ is the projection of $b$ onto $span(\hat{U}) = span(A)$. 

In the underdetermined case, this makes sense because the only way to get a solution of $Ax=b$ is if $b$ is in the column space (or the span) of $A$. If $b$ doesn't appear as a column of $A$, there are plenty of columns to create a linear combination of the different columns to find $b$.

In the overdetermined case, the only way to get a solution is if $b$ is a columnn of $A$. Since there are more rows than column, it is much more likely that there is one component of $b$ that can't be found using a linear combination of the columns of $A$. So, the only way to guarantee a solution is if $b$ is a column of $A$. But by using the SVD, we get $\hat{U}\hat{U}^Tb$ which is a project of $b$ onto the span of $\hat{U}$ (or of $A$), which will determine a solution.