/** Reveal a focused subject in local scrollers without moving the document. */
export function revealWithinScrollRegions(element: HTMLElement): void {
  for (let region = element.parentElement; region && region !== document.scrollingElement; region = region.parentElement) {
    const style = getComputedStyle(region);
    const target = element.getBoundingClientRect();
    const bounds = region.getBoundingClientRect();
    if (['auto', 'scroll'].includes(style.overflowY)) {
      if (target.top < bounds.top || target.height > region.clientHeight) region.scrollTop += target.top - bounds.top;
      else if (target.bottom > bounds.bottom) region.scrollTop += target.bottom - bounds.bottom;
    }
    if (['auto', 'scroll'].includes(style.overflowX)) {
      if (target.left < bounds.left || target.width > region.clientWidth) region.scrollLeft += target.left - bounds.left;
      else if (target.right > bounds.right) region.scrollLeft += target.right - bounds.right;
    }
  }
}
