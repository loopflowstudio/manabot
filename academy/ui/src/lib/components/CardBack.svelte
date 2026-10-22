<script lang="ts">
  // Hidden cards wear the classic card back (fetched once — see
  // scripts/fetch-card-art.mjs). The cache is known to the bundler, so a
  // missing back is a lookup miss, never a 404: the violet material
  // stands in. Like all plates, the back is a fixed world.
  const BACK_URLS = import.meta.glob('/src/lib/card-art/card-back.jpg', {
    eager: true,
    query: '?url',
    import: 'default',
  }) as Record<string, string>;
  const backUrl = Object.values(BACK_URLS)[0] ?? null;

  const MATERIAL =
    'radial-gradient(circle at 50% 42%, rgb(154 142 190 / 0.22) 0, transparent 46%), ' +
    'linear-gradient(152deg, #2a2535 0%, #3d3652 56%, #262130 100%)';
  const background = backUrl
    ? `url('${backUrl}') center / cover no-repeat, ${MATERIAL}`
    : MATERIAL;

  interface Props {
    focused?: boolean;
    clickable?: boolean;
    className?: string;
  }

  let { focused = false, clickable = false, className = '' }: Props = $props();
</script>

<div
  class={`back aspect-[5/7] w-20 overflow-hidden rounded-lg shadow ${focused ? 'ring-1 ring-action' : ''} ${clickable ? 'cursor-pointer' : ''} ${className}`}
  style:background={background}
></div>

<style>
  .back {
    /* A fixed world keeps a fixed frame: no adaptive tokens inside. */
    border: 1px solid rgb(6 9 15 / 0.55);
  }
</style>
