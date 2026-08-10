\begin{document}


\maketitle


\begin{abstract}
  [One paragraph. State the question (what a single training example causally does to a model), the design in one sentence (32 models, 4 conditions x 8 seeds, one 194-token passage written into one row of one batch at step 200), the three findings in order (the intervention registers strongly where it lands; it produces no content-specific difference at the end; it displaces the weights substantially without moving the model out of its basin), and the one-line implication for attribution. No citations.]
\end{abstract}


\section{Introduction}
\label{sec:intro}

[Paragraph 1 --- the problem. Data attribution estimates a counterfactual: what this model would be if one training example had been absent. That counterfactual is almost never measured directly, because removing an example from a real training run changes everything downstream of it.]

[Paragraph 2 --- the gap. Ground-truth single-example counterfactuals at pretraining scale are scarce. Name what is needed to produce one: a run in which everything except the single example is held bit-identical, and a matched control that differs only in that example's absence.]

[Paragraph 3 --- what we do, in three sentences. The design, the fact that the outcome measure and the analysis were fixed before any model was trained, and the fact that the effect is measured both where the example lands and 9,336 steps later.]

[Paragraph 4 --- contributions, as a short inline list. (i) a directly measured single-example counterfactual at pretraining scale, verified bit-identical up to the injection step; (ii) a pre-registered null on whether the example's content changes the resulting model; (iii) a geometric account of what the example does change --- large weight displacement, unchanged basin; (iv) the trajectory of the disturbance across the remaining steps.]


\section{Setup}
\label{sec:setup}

[Paragraph --- training configuration: model family and size, corpus, total steps, precision, optimiser, and the determinism settings that make the counterfactual exact. Point to Table~\ref{tab:conditions}.]

\begin{table}[h]
  \caption{[Caption: state that the four conditions differ only in what text is written into a single row of a single batch at step 200, that every run sharing a seed is bit-identical before that step, and that each condition was run at 8 seeds. Take-away: the only variable in the study is the content of one training example.]}
  \label{tab:conditions}
  \centering
  \begin{tabular}{llc}
    \toprule
    Condition & Text written into one row at step 200 & Runs \\
    \midrule
    \texttt{fluent-true}  & Grammatical English; the assertion is true   & 8 \\
    \texttt{fluent-false} & Grammatical English; the assertion is false  & 8 \\
    \texttt{random-chars} & Random characters; no word structure         & 8 \\
    \texttt{control}      & Nothing written (matched control)            & 8 \\
    \bottomrule
  \end{tabular}
\end{table}

\subsection{Stimuli and gradient matching}

[Paragraph --- the two fluent passages: same register, same structure, exactly equal token length. State that they are matched on the gradient contribution that actually reaches the optimiser rather than on a proxy, and point to Table~\ref{tab:matching}.]

[Paragraph --- the confound, stated plainly. The true passage's subject occurs in the training corpus and the false passage's subject does not, so truth and corpus attestation vary together. State that this was measured before any run existed, and that it is difficult to avoid in principle.]

\begin{table}[h]
  \caption{[Caption: explain that column 2 is the norm of the change the injected row makes to the full-batch gradient at the injection step, column 3 expresses that change as a fraction of the full gradient, and column 4 is the cosine between each condition's gradient change and \texttt{fluent-false}'s. Take-away: the two fluent conditions are matched to 0.14\% in magnitude and point in similar directions, while \texttt{random-chars} is larger and points elsewhere.]}
  \label{tab:matching}
  \centering
  \begin{tabular}{lccc}
    \toprule
    Condition & Gradient change $\|\Delta g\|$ & As \% of full gradient & Cosine with \texttt{fluent-false} \\
    \midrule
    \texttt{fluent-false} & 0.010698 & 2.23 & 1.000 \\
    \texttt{fluent-true}  & 0.010683 & 2.22 & 0.945 \\
    \texttt{random-chars} & 0.011788 & 2.46 & 0.790 \\
    \bottomrule
  \end{tabular}
\end{table}

\subsection{What we measure}

[Paragraph --- define the two endpoint measures, one or two sentences each: the loss barrier along the linear interpolation between a run's final weights and its seed-matched control's, and held-out cross-entropy. Then define the reference scale: the same barrier computed between controls of two different seeds, which is the variation produced by changing the seed and nothing else.]

[Sentence --- state that the outcome measure, the single confirmatory comparison, and the correction policy were fixed before any model existed.]


\section{Does the injected example register at all?}
\label{sec:exp1}

\paragraph{What we wanted to know.} [One or two sentences: whether a single row in a 256-row batch produces any measurable effect at the moment it enters, and whether the conditions differ from one another there. This determines whether a later null is a real null or a failed intervention.]

\paragraph{What we did.} [Two or three sentences: per-step training loss is recorded for every run, and runs sharing a seed consume an identical data order, so the loss at a given step compares models on the same batch. Compare each condition against its seed-matched control at the injection step, paired within seed.]

\paragraph{What we found.} [Two or three sentences pointing at Table~\ref{tab:injection}: which step is the first to differ, how the conditions order, and how large and how consistent the true-versus-false difference is.]

\begin{table}[h]
  \caption{[Caption: explain that each row compares one condition against its seed-matched control on the same batch at the injection step, that differences are paired within seed and averaged over 8 seeds, and that the last column counts how many seeds agree in sign. Take-away: the intervention is detectable where it lands, the conditions order as the design predicts, and the true/false difference holds in every seed.]}
  \label{tab:injection}
  \centering
  \begin{tabular}{lcccc}
    \toprule
    Comparison & Mean loss difference & $t(7)$ & $p$ & Seeds agreeing in sign \\
    \midrule
    \texttt{random-chars} vs.\ control & $+0.00108$ & $+3.22$ & $0.015$ & 6 of 8 \\
    \texttt{fluent-false} vs.\ control & $+0.00024$ & $+0.71$ & $0.503$ & 5 of 8 \\
    \texttt{fluent-true} vs.\ control  & $+0.00006$ & $+0.19$ & $0.858$ & 4 of 8 \\
    \midrule
    \texttt{fluent-false} vs.\ \texttt{fluent-true} & $+0.000175$ & $+13.92$ & $2.3\times10^{-6}$ & 8 of 8 \\
    \bottomrule
  \end{tabular}
\end{table}

\paragraph{What we learned.} [Two sentences: the intervention is real and behaves as the design predicts, so any null at the endpoint is a null result rather than a failed manipulation. Add the caveat that this quantity is a property of the stimuli under a fixed model, not an outcome of training.]


\section{Does the content of the example change the resulting model?}
\label{sec:exp2}

\paragraph{What we wanted to know.} [One or two sentences: the pre-registered question. Holding form, length, and injected gradient magnitude fixed, does the truth of the assertion change how far the single step displaces the model?]

\paragraph{What we did.} [Two or three sentences: compute each condition's displacement from its seed-matched control on both endpoint measures, difference the two fluent conditions within seed, and test against zero across 8 seeds. State that a single confirmatory comparison was registered, so no multiple-comparison correction applies.]

\paragraph{What we found.} [Two or three sentences pointing at Tables~\ref{tab:contrast} and~\ref{tab:displacement}: the registered comparison is null on both measures, and separately, all three injecting conditions displace the model by indistinguishable amounts.]

\begin{table}[h]
  \caption{[Caption: explain that each row is the pre-registered comparison of the true against the false passage on one measure, differenced within seed and tested against zero across 8 seeds. Take-away: neither measure detects an effect of the assertion's truth.]}
  \label{tab:contrast}
  \centering
  \begin{tabular}{lccccc}
    \toprule
    Measure & Mean difference & $t(7)$ & $p$ & 95\% CI & Seeds agreeing in sign \\
    \midrule
    Loss barrier  & $+0.0068$  & $+0.70$ & $0.509$ & $[-0.0085,\ +0.0262]$   & 5 of 8 \\
    Held-out loss & $-0.00044$ & $-1.10$ & $0.310$ & $[-0.00122,\ +0.00024]$ & 4 of 8 \\
    \bottomrule
  \end{tabular}
\end{table}

\begin{table}[h]
  \caption{[Caption: explain that the first three rows give each condition's displacement from its own seed-matched control, and the final row gives the same quantity measured between controls of two different seeds --- the variation produced by changing the seed alone. Take-away: the three conditions are indistinguishable from one another, and every one of them sits far below the seed-alone scale.]}
  \label{tab:displacement}
  \centering
  \begin{tabular}{lcc}
    \toprule
    Comparison & Loss barrier (mean $\pm$ sd) & Weight distance (L2) \\
    \midrule
    \texttt{fluent-false} vs.\ its control & $0.153 \pm 0.016$ & $236.9$ \\
    \texttt{fluent-true} vs.\ its control  & $0.146 \pm 0.024$ & $233.4$ \\
    \texttt{random-chars} vs.\ its control & $0.144 \pm 0.025$ & $234.8$ \\
    \midrule
    Control vs.\ control, different seeds  & $4.930 \pm 0.36$  & $532.9$ \\
    \bottomrule
  \end{tabular}
\end{table}

\paragraph{What we learned.} [Two or three sentences: state the null in the terms it licenses, note that random characters move the model as far as grammatical English does, and state explicitly what the null does not license, given one stimulus pair and the attestation confound.]


\section{What does the example change, if not the outcome?}
\label{sec:exp3}

\paragraph{What we wanted to know.} [One or two sentences: given that the conditions are indistinguishable, whether the intervention changed anything measurable at all, and how large that change is against the seed-alone scale.]

\paragraph{What we did.} [Two sentences: compare each injected run to its control on two measures that respond to different things --- Euclidean distance in weight space, and the loss barrier along the interpolation between them --- and express each as a fraction of the corresponding seed-alone quantity.]

\paragraph{What we found.} [Two or three sentences pointing at Table~\ref{tab:geometry} and Figure~\ref{fig:interpolation}: the two measures disagree by more than an order of magnitude, and the barrier values for injected runs and for seed changes do not overlap.]

\begin{table}[h]
  \caption{[Caption: explain that column 2 compares an injected run against its own control, column 3 compares two controls of different seeds, and column 4 expresses column 2 as a percentage of column 3. Take-away: measured by distance the intervention is a large fraction of a seed change; measured by the barrier it is a small one, so the example moves the model far without moving it into a different basin.]}
  \label{tab:geometry}
  \centering
  \begin{tabular}{lccc}
    \toprule
    Measure & Effect of the injected row & Effect of changing the seed & Ratio \\
    \midrule
    Weight distance (L2) & $235.0$ & $532.9$ & $44\%$  \\
    Loss barrier         & $0.148$ & $4.930$ & $3.0\%$ \\
    \bottomrule
  \end{tabular}
\end{table}

\begin{figure}[h]
  \centering
  \fbox{\rule[-.5cm]{0cm}{4cm} \rule[-.5cm]{10cm}{0cm}}
  \caption{[Figure: loss along the linear interpolation between two finished models, averaged over pairs, with two curves on shared axes --- one for an injected run against its own control, one for two controls of different seeds. X axis from 0 to 1, Y axis in nats. Caption should explain that a rise above the endpoints means the two models cannot be mixed without a loss penalty. Take-away: the seed-change curve rises to a broken model, while the injected-row curve is nearly flat.]}
  \label{fig:interpolation}
\end{figure}

\paragraph{What we learned.} [Two or three sentences: the intervention produces a large but basin-preserving displacement. State why that distinction matters for reading the null in Section~\ref{sec:exp2}, and what it implies about which measures of single-example influence will register an effect and which will not.]


\section{Where does the disturbance go over training?}
\label{sec:exp4}

\paragraph{What we wanted to know.} [One or two sentences: whether the effect of the injected row grows, decays, or persists over the remaining steps, and whether the conditions separate anywhere along the way even though they do not at the end.]

\paragraph{What we did.} [Two sentences: because runs sharing a seed share a data order, the per-step loss difference from the control is a clean divergence measure. Track it from the injection step to the end of training, averaged across seeds.]

\paragraph{What we found.} [Two or three sentences pointing at Figure~\ref{fig:trajectory} and Table~\ref{tab:trajectory}: a latency, then rapid amplification to a peak, then decay to a plateau above the starting level --- and all three conditions following the same curve.]

\begin{figure}[h]
  \centering
  \fbox{\rule[-.5cm]{0cm}{4cm} \rule[-.5cm]{10cm}{0cm}}
  \caption{[Figure: mean absolute per-step loss difference from the seed-matched control, log--log, from the injection step to the end of training, one line per injecting condition. Take-away: the three conditions are indistinguishable at every point along the curve.]}
  \label{fig:trajectory}
\end{figure}

\begin{table}[h]
  \caption{[Caption: explain that each row is a training step, the value is the mean absolute per-step loss difference from the seed-matched control averaged over 8 seeds, and the final column expresses that value relative to the step immediately after injection. Take-away: the disturbance grows by two orders of magnitude before decaying, and settles well above where it began.]}
  \label{tab:trajectory}
  \centering
  \begin{tabular}{lcc}
    \toprule
    Training step & Mean difference from control & Relative to step 201 \\
    \midrule
    201  & $0.000080$ & $1\times$   \\
    210  & $0.000066$ & $0.8\times$ \\
    240  & $0.001581$ & $20\times$  \\
    260  & $0.010296$ & $128\times$ \\
    500  & $0.004886$ & $61\times$  \\
    2000 & $0.001623$ & $20\times$  \\
    9535 & $0.001426$ & $18\times$  \\
    \bottomrule
  \end{tabular}
\end{table}

\paragraph{What we learned.} [Two or three sentences: the disturbance is amplified by training dynamics rather than retained as content, which accounts for the endpoint result. Note the consequence for attribution methods that assume a single example's influence decays monotonically, or persists in proportion to what the example said.]


\section{Limitations}
\label{sec:limitations}

[Compact paragraph or short list, in order of severity: one stimulus pair; truth entangled with corpus attestation; one model size, one injection step, one passage length, one position in the batch; 8 seeds against a planned 10 and the reason the study stopped; the barrier is a lower bound computed on a finite grid; and the seed-alone reference is a conservative scale rather than the null distribution of the tested quantity.]


\section{Discussion}
\label{sec:discussion}

[Paragraph 1 --- what this measurement contributes: a directly measured single-example counterfactual at pretraining scale, with the confounds held fixed rather than modelled.]

[Paragraph 2 --- what the disagreement between the two measures implies for how single-example influence should be quantified.]

[Paragraph 3 --- what would have to change to detect a content effect if one exists: more stimulus pairs, larger or repeated interventions, earlier injection, or measures targeted at the injected content rather than at global loss.]


\begin{ack}
[Funding and competing interests. Omitted from the anonymized submission.]
\end{ack}


\section*{References}

[No citations in this draft.]


\appendix

\section{Reproducibility and verification}
\label{app:repro}

[Placeholder --- the checks establishing that the counterfactual is exact: all conditions bit-identical at the last step before injection, across both training machines; one corpus verified byte-identical across machines; the injected text verified against its recorded digest in every run; no interrupted runs. Include the digest table.]

\section{Robustness of the evaluation}
\label{app:robustness}

[Placeholder --- the evaluation-grid check at four times the number of held-out windows, and the independent recomputation of a previously published quantity on different hardware. Include both comparison tables.]

\section{Pre-registration and departures}
\label{app:prereg}

[Placeholder --- what was fixed in advance and on what date, and each departure: the second registered comparison whose condition was removed before the runs, the seed count against the planned one and why the study stopped, and the treatment of the reference scale for a pairwise measure.]


\end{document}
