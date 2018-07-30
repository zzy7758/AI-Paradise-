#  Universal  Approximation Theorem

通用近似定理(universal approximation theorem)表明,一个仅有单隐藏层的前馈神经网络，如果它具有足够多的神经元和非线性的激活函数，**有能力以任意精度拟合任意一个从有限维空间映射到另一个有限维空间的Borel可测函数**。对于神经网络，我们只要知道定义在的有界闭集上的任意连续函数都是Borel可测的，因此可以使用神经网络进行近似。

## 通用近似定理的限制

１．通用近似定理意味着无论我们试图学习什么样的函数，我们知道一个足够大的神经网络肯定**有能力表示**这个函数。然而，我们并不能保证我们能够**学习到**这个函数。

这主要由两个原因造成：

- 用于训练的优化算法可能找不到用于期望函数的参数值，
- 训练算法可能由于过拟合而选择了错误的函数

用我自己的说法来说的话，应该是两个方面的原因：

- 数据方面的原因，我们并没有足够的数据，用于训练的数据可能在函数映射的时候，并不完全
- 优化算法的原因，过拟合或者欠拟合，或者干脆是当前的优化算法无法找到函数的最优解，因为它总是找到局部最小值

２．通用近似定理说明，存在一个足够大的网络用任意精度表示目标函数，但是并没有说明这个网络需要有多大。当前的证明给出了指数维的隐藏单元数量，而这是不切实际的。所以，理论与实践的之间还是有很大的差距。

其实对于现实问题来说，有时候我们根本不知道我们想要的函数是什么样子的。

总之，具有隐层的前馈神经网络已经具有强大的表示能力，但是由于现实条件的限制，这并不能形成一个实际的指导准则来指导模型的选择和训练。但是在大多数情况下，使用**更深的**模型能够减少表示期望函数需要的神经元数量，并且可以减少学习误差。

## 深度over 参数数量

存在一些函数能够在网络的深度大于某个值的时候被高效的近似，而在网络深度较小的时候，需要一个远远大于之前的模型。在很多情况下，浅层模型所需要的隐藏单元个数是指数级的。

深度　vs 参数数量

- 理论证据－Montufar 的主要定理指出，深度整流网络可以描述的线性区域的数量是深度的指数级。
- 统计原因－
- 假设原因－选择深度模型默许了一个非常普遍的信念，那就是我们想要学的函数应该涉及几个更加简单的函数的组合，这可以用表示学习的观点来解释，我们相信学习的问题包含发现一组潜在的变差因素，他们可以根据其他简单的潜在的变差因素来描述。或者，我们可以将深度结构的使用解释为另一种信念，那就是我们想要学习的函数是包含多个步骤的程序，其中每个步骤使用前一个步骤的输出。
- evidence－实验结果表明，更深层的网络能够更好的泛化，测试集上的准确率随着深度的增加而不断增加，而模型的其他尺寸并没有这个效果。

## 相关论文

### 最早的证明Approximation by Superpositions of a Sigmoidal Function

n this paper we demonstrate that finite linear combinations of  compositions of a fixed, univariate function and a set of affine  functionals can uniformly approximate any continuous function of n real  variables with support in the unit hypercube; only mild conditions are  imposed on the univariate function. Our results settle an open question  about representability in the class of single bidden layer neural  networks. In particular, we show that arbitrary decision regions can be  arbitrarily well approximated by continuous feedforward neural networks  with only a single internal, hidden layer and any continuous sigmoidal  nonlinearity. The paper discusses approximation properties of other  possible types of nonlinearities that might be implemented by artificial  neural network 

这个算法的证明需要用到很多数学知识（包括：Hahn-Banach 理论, Riesz Representation 理论, Fourier  analysis），所以实际上他不容易理解。本文也不采用直接的数学证明的方式来证明这个定理，因为对于更多情况下，我们只需要直观理解足够了，为了方便有兴趣的读者阅读严谨的数学证明，以下是原论文链接：   <http://www.dartmouth.edu/~gvc/Cybenko_MCSS.pdf>

### 另外一篇重要的相关文章

Uses the Stone-Weierstrass theorem and the cosine squasher of A. R.  Gallant and H. White (1988) to establish that standard multilayer  feedforward networks  that **use arbitrary squashing functions** are  capable of approximating any measurable function to any desired degree  of accuracy.　 MFNs are therefore a class of universal approximators.  Results provide a basis for establishing the ability of MFNs to learn  (i.e., estimate consistently) connection strengths of approximations.  

嗯，这就是每个字我都懂，但是在一起的时候，我就感动的要哭。这篇文章的结论是对上面一篇文章的扩展，因为它的要求从sigmoid函数扩展到了任何squashing 函数了。

[Multilayer  feedforward networks are universal approximators](http://www.sciencedirect.com/science/article/pii/0893608089900208), 

### 通用近似定理的可视化理解

另外Michael Nielson的教程中，使用另一种方式，也提供了可视化的证明过程：   <http://neuralnetworksanddeeplearning.com/chap4.html>。   网站使用了一些js脚本，使得我们可以直接操作整个神经元的优化过程，非常有趣

#### [Conclusion](http://neuralnetworksanddeeplearning.com/chap4.html#conclusion)

虽然这个结果并不能直接被应用于构建神经网络，它还是非常重要，因为它解决了一个问题，就是一个确定的函数能不能被神经网络来表示。这个答案总是"yes"。所以在神经网络中正确的问题是：怎样用一个更好的方法来使用神经网络表示一个目标函数？

这里我们开发的的证明示例只使用了两个隐层来表示任何函数。此外，正如我们曾经讨论的，其实使用具有一个隐层的神经网络剧就可以得到相同的结果了。尽管如此，你可能想知道为什么我们还对深层网络有兴趣呢？我们不能用浅层的或者只有一层的神经网络么？

虽然在原理上是可以的，但是有非常实际的原因让我们使用深层网络。就像在第一章中说的，深度网络的层次结构让它们能够适应的学习知识的层次，这貌似对解决现实世界的问题非常有用处。更具体的说，当想要解决类似于图片识别等问题的时候，使用一个不仅仅能够理解么一个像素的，并且渐渐理解更加复杂概念：从边(edge)到简单的几何形状，直到最复杂的多对象识别是很有帮助的。在后面的章节中，我们将会看到证据证明深层的网络比浅层的网络在学习这种知识的层次结构的时候更有优势。　

总的来说：通用近似定理告诉我们，神经网络可以计算所有的函数，并且实践证明显示深层网络更适合于学习解决现实世界问题的函数，也就是说更加具有实际意义，因为更深的模型能够减少需要的表示特定函数的神经元的数量，并且可以减少泛化误差。

Although the result isn't directly useful in constructing networks, it's important because it takes off the table the question of whether any particular function is computable using a neural network.  The answer to that question is always "yes".  So the right question to ask is not whether any particular function is computable, but rather what's a *good* way to compute the function.

The universality construction we've developed uses just two hidden layers to compute an arbitrary function.  Furthermore, as we've discussed, it's possible to get the same result with just a single hidden layer.  Given this, you might wonder why we would ever be interested in deep networks, i.e., networks with many hidden layers. Can't we simply replace those networks with shallow, single hidden layer networks?

While in principle that's possible, there are good practical reasons to use deep networks.  As argued in [Chapter 1](http://neuralnetworksanddeeplearning.com/chap1.html#toward_deep_learning), deep networks have a hierarchical structure which makes them particularly well adapted to learn the hierarchies of knowledge that seem to be useful in solving real-world problems.  Put more concretely, when attacking problems such as image recognition, it helps to use a system that understands not just individual pixels, but also increasingly more complex concepts: from edges to simple geometric shapes, all the way up through complex, multi-object scenes. In later chapters, we'll see evidence suggesting that deep networks do a better job than shallow networks at learning such hierarchies of knowledge.  To sum up: universality tells us that neural networks can compute any function; and empirical evidence suggests that deep networks are the networks best adapted to learn the functions useful in solving many real-world problems.