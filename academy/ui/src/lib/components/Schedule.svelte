<script lang="ts">
  import type { CombatView, ScheduleView } from '$lib/types';

  let { schedule, heroSeat, heroLabel, villainLabel }: {
    schedule?: ScheduleView;
    heroSeat: number;
    heroLabel: string;
    villainLabel: string;
  } = $props();

  function player(seat: number): string {
    return seat === heroSeat ? heroLabel : villainLabel;
  }

  function restriction(combat: CombatView): string {
    return combat.attackers
      ? `Only ${combat.attackers} can attack.`
      : 'No additional attacker restriction.';
  }
</script>

<section aria-label="Turn and combat schedule" class="type-caption px-3 py-2" data-testid="schedule">
  {#if !schedule}
    <p>Schedule not recorded.</p>
  {:else}
    {#if schedule.extra_turns.length || schedule.resume_with !== undefined}
      <p>After the current turn:</p>
      <ol aria-label="Scheduled turns" class="list-decimal pl-5">
        {#each schedule.extra_turns as seat}
          <li>{player(seat)} — extra turn</li>
        {/each}
        {#if schedule.resume_with !== undefined}
          <li>{player(schedule.resume_with)} — normal turn resumes</li>
        {/if}
      </ol>
    {:else}
      <p>No extra turns scheduled.</p>
    {/if}
    {#if schedule.current_combat}
      <p data-testid="current-combat">Current combat — {player(schedule.current_combat.player_index)}.
        {restriction(schedule.current_combat)}</p>
    {/if}
    {#if schedule.additional_combats.length}
      <p>Extra combats still to come this turn:</p>
      <ol aria-label="Scheduled extra combats" class="list-decimal pl-5">
        {#each schedule.additional_combats as combat}
          <li>{player(combat.player_index)}. {restriction(combat)}</li>
        {/each}
      </ol>
    {:else}
      <p>No extra combats pending.</p>
    {/if}
  {/if}
</section>
