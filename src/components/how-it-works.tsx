import { NodeDetail } from "@/components/node-detail";
import { weightCount } from "@/lib/mlp";

export function HowItWorks() {
  const { weights } = weightCount();
  return (
    <section className="mt-12 flex flex-col gap-4 border-t border-line pt-8 lg:flow-root">
      <div className="order-3 lg:float-right lg:mb-6 lg:ml-10 lg:w-[min(36rem,48%)]">
        <NodeDetail />
      </div>
      <h2 className="order-1 text-balance text-xl font-medium tracking-tight">How this works</h2>
      <div className="order-2 space-y-4 text-base leading-relaxed text-pretty lg:mt-4">
            <p>
              The grid is a 20-dimensional vector, read left to right, top to bottom. This can be thought
              of like a list of 20 input numbers, though in this simple case the inputs are always 0 or
              1. An empty cell is 0 and a black cell is 1.
            </p>
            <p>This simple neural network has three layers which are each relatively small.</p>
            <p>
              The first layer is just the twenty inputs. (Its neurons don’t have their own separate bias
              values, but you could think of the inputs themselves as being like biases that change.) The
              connections from this layer to the next one all have fixed <em>weights</em> that were
              learned in <em>training</em> the network.
            </p>
            <p>
              The second layer is called the <em>hidden layer</em>. It contains eight neurons called{" "}
              <em>hidden units</em>. A smaller number is used here to keep the net smaller but a larger
              number of units would actually work better.
            </p>
            <p>
              The third layer is the <em>output layer</em> which, in this case, has eleven neurons called{" "}
              <em>output units</em> — one for each of the digits 0–9 and ?.
            </p>
            <p>
              Each hairline connection is one <em>weight</em>. There are {weights} of them in total.
              Weights are <em>multiplied</em> by their inputs. They can be negative or positive, with an
              absolute value smaller than 1 or larger than 1. So they can either magnify or diminish
              their inputs, or flip positive inputs into “inhibitory” negative ones. (For color coding:
              gray is zero. The color coded scale reaches pure red at the most negative value and pure
              green at the most positive. The maximum and minimum values among the weights and biases
              were learned in training.)
            </p>
            <p>After multiplying all the inputs by their weights, they are all summed together.</p>
            <p>
              Each neuron except for the input units has a <em>bias</em>, which is kind of like a weight
              except that it is <em>added</em> to the sum of all the inputs. (The bias is shown in the
              diagram using the same color coding as the weights in the fill color of the neuron.)
            </p>
            <p>
              Finally, the value is passed through a <em>transfer function</em> that acts sort of like a
              cap on the minimum and maximum output value of the neuron. In this case the transfer
              function is <em>tanh()</em>. So the final output of the neuron (unit) is tanh(sum of inputs
              × weights + bias). This function has the effect of “squishing” numbers greater than 0.5 or
              less than −0.5, which makes it harder to get a result close to −1 or 1.
            </p>
            <p>
              The output vector is run through the softmax() function, so the eleven output numbers under
              the output units are scaled such that they’re non-negative and all sum to 1. This converts
              them into normalized probabilities between 0 and 1 where the sum of all the probabilities
              can’t be higher than 100%.
            </p>
            <p>
              The large “recognized” numeral is the class (output neuron) with the highest probability
              given the numeral you drew in the grid, and thus the input vector to the neural net.
            </p>
            <p>
              Clear the grid and the blue disappears. Every weight is multiplied by 0, each hidden unit
              equals tanh() of its bias, and the softmax is normalizing biases alone. That empty (all
              0’s) input vector was trained to come out as ? with a probability near 1.
            </p>
          </div>
    </section>
  );
}
