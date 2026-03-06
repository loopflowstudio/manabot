import type { GameLogEntry, ReplayFrame, Trace } from './types';

export function buildReplayFrames(trace: Trace): ReplayFrame[] {
  if (trace.events.length === 0) {
    return [
      {
        observation: trace.final_observation,
        actionDescription: null,
        actor: null,
      },
    ];
  }

  const frames: ReplayFrame[] = [
    {
      observation: trace.events[0].observation,
      actionDescription: null,
      actor: null,
    },
  ];

  for (let index = 0; index < trace.events.length; index += 1) {
    const event = trace.events[index];
    const nextObservation =
      trace.events[index + 1]?.observation ?? trace.final_observation;

    frames.push({
      observation: nextObservation,
      actionDescription: event.action_description,
      actor: event.actor,
    });
  }

  return frames;
}

export function replayLogEntries(trace: Trace): GameLogEntry[] {
  return trace.events.map((event, index) => ({
    id: `replay-${index}`,
    actor: event.actor,
    text: `${event.actor === 'hero' ? 'Hero' : 'Villain'}: ${event.action_description}`,
  }));
}
