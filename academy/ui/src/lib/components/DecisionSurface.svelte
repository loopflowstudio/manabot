<script lang="ts">
  import { subjectLabel } from '$lib/decision-labels';
  import ActionPanel from './ActionPanel.svelte';
  import DecisionContext from './DecisionContext.svelte';
  import { offerFocus } from '$lib/offer-map';
  import type { ExperienceFrame, InteractionOffer } from '$lib/types';

  interface Props {
    frame: ExperienceFrame;
    navigationKey?: string;
    selectedTargetId?: number | null;
    disabled?: boolean;
    readOnly?: boolean;
    chosenOffer?: number | null;
    fastForwarding?: boolean;
    canPassTurn?: boolean;
    highlightedOfferIds?: Set<number>;
    onHoverAction?: (offer: InteractionOffer | null) => void;
    onSelectAction?: (offer: InteractionOffer) => void;
    onClearSelection?: () => void;
    onPassTurn?: () => void;
  }
  let { frame, navigationKey = '', selectedTargetId = null, disabled = false, readOnly = false,
    chosenOffer = null, fastForwarding = false, canPassTurn = false,
    highlightedOfferIds = new Set<number>(), onHoverAction, onSelectAction,
    onClearSelection, onPassTurn }: Props = $props();
  const context = $derived(frame.projection.decision_context);
  const offers = $derived(frame.action_space === 'LEARN' || selectedTargetId === null ? frame.offers :
    frame.offers.filter(offer => offerFocus(offer).includes(selectedTargetId!)));
  const previewNames = $derived(Object.fromEntries([
    ...frame.projection.agent.hand.map(card => [card.id, card.name]),
    ...(frame.projection.agent.inspected ?? []).map(card => [card.id, card.name]),
    ...(frame.projection.agent.sideboard ?? []).map(card => [card.candidate_id, card.name]),
  ]));
</script>

<section aria-label="Decision surface" data-testid="decision-surface" data-revision={frame.revision} class="min-w-0 space-y-3">
  <ActionPanel {offers} describeSubject={object => subjectLabel(frame.projection, object)} {previewNames} actionSpaceKind={frame.action_space}
    instruction={context?.instruction ?? frame.prompt?.instruction ?? ''}
    totalOffers={frame.offers.length} {chosenOffer} {selectedTargetId}
    {disabled} {readOnly} {fastForwarding} {canPassTurn} {highlightedOfferIds}
    focusKey={`${frame.match_id}:${frame.revision}:${frame.prompt?.id}:${navigationKey}:${selectedTargetId ?? 'all'}`}
    {onHoverAction} {onSelectAction} {onClearSelection} {onPassTurn} />
  <DecisionContext observation={frame.projection} actor={frame.prompt?.actor ?? null} />
</section>
