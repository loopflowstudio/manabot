"""Render retained learning curves and descriptive score/time tradeoffs.

Every plotted value is a measured checkpoint. Lines connect observations only;
they do not imply a fixed-wall-time experiment or measured intermediate scores.
The Pareto set preserves both anchors and the worst observed training seed.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pydantic import BaseModel

from learning_selection import Point

type Clock = Literal['transitions', 'inclusive', 'active']
ORDER = ('small','small-deep','middle','large-shallow','large')
LABELS = {'small':'64×2','small-deep':'64×4','middle':'192×4',
          'large-shallow':'384×2','large':'384×8'}
COLORS = {4:'#626262',64:'#0072b2',128:'#009e73',256:'#d55e00',512:'#cc79a7'}
STYLES = ('solid','dashed','dotted')
MARKERS = ('o','s','^')


class Tradeoff(BaseModel):
    cell: str
    size: str
    streams: int
    transitions: int
    seed_greedy_scores: dict[int,float]
    seed_random_scores: dict[int,float]
    seed_inclusive_seconds: dict[int,float]
    seed_active_seconds: dict[int,float]
    worst_seed_greedy_score: float
    worst_seed_random_score: float
    worst_seed_inclusive_seconds: float
    worst_seed_active_seconds: float
    pareto: bool = False


def mark_frontier(records: Sequence[Tradeoff]) -> list[Tradeoff]:
    """Minimize inclusive time and maximize both anchor scores, without averaging seeds."""
    def dominates(left: Tradeoff, right: Tradeoff) -> bool:
        no_worse = (left.worst_seed_inclusive_seconds <= right.worst_seed_inclusive_seconds
                    and left.worst_seed_greedy_score >= right.worst_seed_greedy_score
                    and left.worst_seed_random_score >= right.worst_seed_random_score)
        strictly_better = (left.worst_seed_inclusive_seconds < right.worst_seed_inclusive_seconds
                           or left.worst_seed_greedy_score > right.worst_seed_greedy_score
                           or left.worst_seed_random_score > right.worst_seed_random_score)
        return no_worse and strictly_better

    return [record.model_copy(update={'pareto':not any(dominates(other,record) for other in records)})
            for record in records]


def build_tradeoffs(points: Sequence[Point]) -> list[Tradeoff]:
    indexed = {(p.cell,p.seed,p.anchor,p.transitions):p for p in points}
    if len(indexed) != len(points):
        raise ValueError('Duplicate score coordinate')
    seeds = sorted({p.seed for p in points})
    cells = sorted({(p.cell,p.transitions) for p in points})
    expected = {(cell,seed,anchor,count) for cell,count in cells for seed in seeds
                for anchor in ('scripted_greedy','random')}
    if not cells or set(indexed) != expected:
        raise ValueError('Tradeoffs require both anchors and all training seeds at each point')
    records: list[Tradeoff] = []
    for cell,count in cells:
        greedy: dict[int,float] = {}
        random: dict[int,float] = {}
        inclusive: dict[int,float] = {}
        active: dict[int,float] = {}
        sizes: set[str] = set()
        stream_counts: set[int] = set()
        for seed in seeds:
            g = indexed[cell,seed,'scripted_greedy',count]
            r = indexed[cell,seed,'random',count]
            if g.coordinates != r.coordinates:
                raise ValueError('Anchor scores have different training coordinates')
            if g.coordinates.active_training_seconds is None:
                raise ValueError('Missing active training clock; cannot infer it from inclusive time')
            greedy[seed],random[seed] = g.mean,r.mean
            inclusive[seed] = g.coordinates.training_seconds
            active[seed] = g.coordinates.active_training_seconds
            sizes.add(g.size)
            stream_counts.add(g.streams)
        if len(sizes) != 1 or len(stream_counts) != 1:
            raise ValueError('Recipe identity differs across training seeds')
        records.append(Tradeoff(cell=cell,size=sizes.pop(),streams=stream_counts.pop(),
            transitions=count,seed_greedy_scores=greedy,seed_random_scores=random,
            seed_inclusive_seconds=inclusive,seed_active_seconds=active,
            worst_seed_greedy_score=min(greedy.values()),worst_seed_random_score=min(random.values()),
            worst_seed_inclusive_seconds=max(inclusive.values()),
            worst_seed_active_seconds=max(active.values())))
    return mark_frontier(records)


def _x_value(point: Point, clock: Clock) -> float:
    if clock == 'transitions':
        return point.transitions/1000
    if clock == 'inclusive':
        return point.coordinates.training_seconds/60
    active = point.coordinates.active_training_seconds
    if active is None:
        raise ValueError('Cannot plot a missing active training clock')
    return active/60


def curves(points: Sequence[Point], *, clock: Clock, title: str, target: Path) -> None:
    """Show each seed separately; only the observed checkpoint means are joined."""
    shapes = [shape for shape in ORDER if any(p.size == shape for p in points)]
    seeds = sorted({p.seed for p in points})
    streams = sorted({p.streams for p in points})
    if len(seeds) > len(STYLES):
        raise ValueError('Figure needs additional explicit styles for more than three seeds')
    fig,axes = plt.subplots(2,len(shapes),figsize=(3.15*len(shapes),7),squeeze=False,
                           sharey=True)
    for col,shape in enumerate(shapes):
        for row,anchor in enumerate(('scripted_greedy','random')):
            axis = axes[row,col]
            for stream_count in streams:
                for seed_index,seed in enumerate(seeds):
                    series = sorted((p for p in points if (p.size,p.streams,p.seed,p.anchor)==
                                     (shape,stream_count,seed,anchor)),key=lambda p:p.transitions)
                    if not series:
                        continue
                    axis.plot([_x_value(p,clock) for p in series],[100*p.mean for p in series],
                        color=COLORS[stream_count],linestyle=STYLES[seed_index],
                        marker=MARKERS[seed_index],linewidth=1.5,markersize=4.5)
            axis.set_ylim(0,100)
            axis.set_xlim(left=0)
            axis.grid(alpha=.2)
            axis.axhline(50 if anchor=='scripted_greedy' else 75,color='#999999',
                        linestyle=':',linewidth=.8,zorder=0)
            axis.set_title(LABELS[shape] if row==0 else '')
            if col==0:
                axis.set_ylabel(('Greedy' if anchor=='scripted_greedy' else 'Random')+' score (%)')
            if row==1:
                axis.set_xlabel({'transitions':'Learner transitions (thousands)',
                                 'inclusive':'Inclusive training minutes','active':'Active training minutes'}[clock])
            if clock=='transitions':
                ticks = sorted({p.transitions/1000 for p in points})
                axis.set_xticks(ticks,[f'{tick:g}' if tick==0 else f'{tick:.0f}' for tick in ticks])
    handles = [Line2D([0],[0],color=COLORS[s],label=f'{s} streams',linewidth=2) for s in streams]
    handles += [Line2D([0],[0],color='#222222',linestyle=STYLES[i],marker=MARKERS[i],
                      label=f'Training seed {seed}') for i,seed in enumerate(seeds)]
    fig.suptitle(title,fontsize=15,y=.98)
    fig.legend(handles=handles,loc='lower center',ncol=min(len(handles),6),bbox_to_anchor=(.5,.035),frameon=False)
    note = 'Lines join measured checkpoints; no intermediate score is inferred.'
    if clock!='transitions':
        note += ' Time-axis ranges differ by model.'
    fig.text(.5,.012,note,ha='center',fontsize=9,color='#444444')
    fig.subplots_adjust(top=.9,bottom=.2,left=.055,right=.99,hspace=.18,wspace=.18)
    target.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(target)
    fig.savefig(target.with_suffix('.png'),dpi=160)
    plt.close(fig)


def endpoint_tradeoffs(records: Sequence[Tradeoff], *, title: str, target: Path) -> None:
    """Place endpoint recipes on a shared time scale, showing the worst observed seed."""
    endpoint = max(record.transitions for record in records)
    selected = [record for record in records if record.transitions==endpoint]
    fig,axes = plt.subplots(1,2,figsize=(13,6))
    markers = dict(zip(ORDER,('o','s','^','D','P'),strict=True))
    for axis,anchor in zip(axes,('greedy','random'),strict=True):
        for record in selected:
            score = record.worst_seed_greedy_score if anchor=='greedy' else record.worst_seed_random_score
            x = record.worst_seed_inclusive_seconds/60
            axis.scatter(x,score*100,color=COLORS[record.streams],marker=markers[record.size],s=65)
        axis.set_xlabel('Worst-seed inclusive training minutes (log scale)')
        axis.set_ylabel(f'Worst-seed {anchor} score (%)')
        axis.set_xscale('log')
        axis.set_ylim(0,103)
        axis.margins(x=.2)
        axis.grid(alpha=.2)
    fig.suptitle(title,fontsize=14)
    stream_handles = [Line2D([0],[0],color=COLORS[s],marker='o',linestyle='none',
        label=f'{s} streams') for s in sorted({r.streams for r in selected})]
    model_handles = [Line2D([0],[0],color='#333333',marker=markers[size],linestyle='none',
        label=LABELS[size]) for size in ORDER if any(r.size==size for r in selected)]
    fig.legend(handles=stream_handles,loc='lower center',ncol=5,bbox_to_anchor=(.5,.11),frameon=False)
    fig.legend(handles=model_handles,loc='lower center',ncol=5,bbox_to_anchor=(.5,.065),frameon=False)
    fig.text(.5,.025,f'Shapes identify width × attention layers. Every point uses {endpoint:,} learner transitions.',
             ha='center',fontsize=9)
    fig.subplots_adjust(top=.86,bottom=.27,left=.07,right=.99,wspace=.25)
    target.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(target)
    fig.savefig(target.with_suffix('.png'),dpi=160)
    plt.close(fig)
