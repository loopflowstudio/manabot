# MTG-162 implementation assumptions

MTG-161 was authoritatively completed before implementation; its recorded merge
commit `1c92d12e4ab718539007422244e0c16bde265042` supplied the dependency result.
After upstream sync the checkout's HEAD is `60fc6abe`; the working design records
its FiLM/learnability implications.
The supplied Academy runtime note is an additional boundary constraint; the
accepted scope remains the 17 sacrifice/landfall cards.

Treasure requires immediate chosen-color mana during payment. The implementation
uses the shared spell/activation payment continuation and resumes demanded and
waterbend payments without a stack-based mana approximation. Fixed-source
production retains its existing owner. Kicker admission includes these mana
sources. Validation rosters remain separate from final product-deck admission.
