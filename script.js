(function () {
  "use strict";

  const header = document.querySelector("[data-header]");
  const menuButton = document.querySelector("[data-menu-button]");
  const navLinks = document.querySelector("[data-nav-links]");
  const languageButtons = Array.from(document.querySelectorAll("[data-language]"));
  const languageStorageKey = "qinyu-homepage-language";
  const textTranslations = Array.from(document.querySelectorAll("[data-en]"))
    .map((element) => ({ element, zh: element.textContent, en: element.getAttribute("data-en") }));
  const translatedAttributes = ["content", "alt", "aria-label", "title", "src", "href"];
  const attributeTranslations = translatedAttributes.flatMap((attribute) =>
    Array.from(document.querySelectorAll(`[data-en-${attribute}]`)).map((element) => ({
      element,
      attribute,
      zh: element.getAttribute(attribute),
      en: element.getAttribute(`data-en-${attribute}`),
    }))
  );

  const applyLanguage = (language, remember = false) => {
    const selected = language === "en" ? "en" : "zh";
    const headerHeight = header ? header.getBoundingClientRect().height : 0;
    const readingElement = remember
      ? document.elementFromPoint(window.innerWidth / 2, headerHeight + 32)
        ?.closest(".project-list article, .content-section, .profile-section, .contact-section")
      : null;
    const readingOffset = readingElement?.getBoundingClientRect().top;

    document.documentElement.lang = selected === "en" ? "en" : "zh-CN";
    textTranslations.forEach(({ element, zh, en }) => {
      element.textContent = selected === "en" ? en : zh;
    });
    attributeTranslations.forEach(({ element, attribute, zh, en }) => {
      element.setAttribute(attribute, selected === "en" ? en : zh);
    });
    languageButtons.forEach((button) => {
      button.setAttribute("aria-pressed", String(button.getAttribute("data-language") === selected));
    });

    if (readingElement) {
      window.scrollBy({ top: readingElement.getBoundingClientRect().top - readingOffset, behavior: "instant" });
    }
    if (remember) {
      try { window.localStorage.setItem(languageStorageKey, selected); } catch (_) { /* Storage can be unavailable in local previews. */ }
      const url = new URL(window.location.href);
      url.searchParams.set("lang", selected);
      try { window.history.replaceState(window.history.state, "", url); } catch (_) { /* Language switching also works for file previews. */ }
    }
  };

  let savedLanguage = null;
  try { savedLanguage = window.localStorage.getItem(languageStorageKey); } catch (_) { /* Default to Chinese without storage. */ }
  const requestedLanguage = new URL(window.location.href).searchParams.get("lang");
  const initialLanguage = ["zh", "en"].includes(requestedLanguage)
    ? requestedLanguage
    : ["zh", "en"].includes(savedLanguage) ? savedLanguage : "zh";
  applyLanguage(initialLanguage);
  languageButtons.forEach((button) => {
    button.addEventListener("click", () => {
      applyLanguage(button.getAttribute("data-language"), true);
      closeMenu();
    });
  });

  const navAnchors = Array.from(document.querySelectorAll('.nav-links a[href^="#"]'));
  const sections = navAnchors
    .map((anchor) => document.querySelector(anchor.getAttribute("href")))
    .filter(Boolean);

  const setHeaderState = () => {
    if (header) header.classList.toggle("is-scrolled", window.scrollY > 12);
  };

  const closeMenu = (returnFocus = false) => {
    if (!menuButton || !navLinks) return;
    menuButton.setAttribute("aria-expanded", "false");
    navLinks.classList.remove("is-open");
    if (returnFocus) menuButton.focus();
  };

  setHeaderState();
  window.addEventListener("scroll", setHeaderState, { passive: true });

  if (menuButton && navLinks) {
    menuButton.addEventListener("click", () => {
      const willOpen = menuButton.getAttribute("aria-expanded") !== "true";
      menuButton.setAttribute("aria-expanded", String(willOpen));
      navLinks.classList.toggle("is-open", willOpen);
    });

    navLinks.addEventListener("click", (event) => {
      if (event.target.closest("a")) closeMenu();
    });

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && menuButton.getAttribute("aria-expanded") === "true") closeMenu(true);
    });

    window.addEventListener("resize", () => {
      if (window.innerWidth > 800) closeMenu();
    });
  }

  const resourceCards = Array.from(document.querySelectorAll("[data-project-resources]"));
  const canHover = window.matchMedia("(hover: hover)");
  let activeResourceCard = null;
  let resourcesPinned = false;
  let waitForResourcePointerMove = false;

  const closeResources = (returnFocus = false) => {
    if (!activeResourceCard) return;
    const card = activeResourceCard;
    card.querySelector("[data-project-options]").hidden = true;
    card.querySelector("[data-project-toggle]").setAttribute("aria-expanded", "false");
    card.classList.remove("is-resource-open");
    activeResourceCard = null;
    resourcesPinned = false;
    if (returnFocus) card.querySelector("[data-project-toggle]").focus({ preventScroll: true });
  };

  const openResources = (card, pin = false) => {
    if (activeResourceCard !== card) closeResources();
    activeResourceCard = card;
    resourcesPinned = resourcesPinned || pin;
    card.querySelector("[data-project-options]").hidden = false;
    card.querySelector("[data-project-toggle]").setAttribute("aria-expanded", "true");
    card.classList.add("is-resource-open");
  };

  const toggleResources = (card) => {
    if (activeResourceCard === card && resourcesPinned) closeResources();
    else openResources(card, true);
  };

  resourceCards.forEach((card) => {
    card.addEventListener("pointerenter", (event) => {
      if (!waitForResourcePointerMove && canHover.matches && event.pointerType !== "touch") openResources(card);
    });
    card.addEventListener("pointerleave", () => {
      if (activeResourceCard === card && !resourcesPinned && !card.contains(document.activeElement)) closeResources();
    });
    card.querySelector("[data-project-toggle]").addEventListener("click", () => toggleResources(card));
    card.addEventListener("click", (event) => {
      if (event.target.closest("a, button, [data-project-options]") || window.getSelection()?.toString()) return;
      toggleResources(card);
    });
    card.querySelector("[data-project-options]").addEventListener("click", (event) => {
      if (event.target.closest("a")) {
        waitForResourcePointerMove = true;
        closeResources(true);
      }
    });
    card.addEventListener("focusout", (event) => {
      if (activeResourceCard === card && !card.contains(event.relatedTarget)) closeResources();
    });
  });

  if (resourceCards.length) {
    document.addEventListener("pointermove", (event) => {
      if (!waitForResourcePointerMove) return;
      waitForResourcePointerMove = false;
      const card = event.target.closest("[data-project-resources]");
      if (card && canHover.matches && event.pointerType !== "touch") openResources(card);
    }, { passive: true });
    document.addEventListener("click", (event) => {
      if (activeResourceCard && !activeResourceCard.contains(event.target)) closeResources();
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && activeResourceCard) {
        event.preventDefault();
        closeResources(true);
      }
    });
  }

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const revealItems = document.querySelectorAll("[data-reveal]");

  if (reducedMotion) {
    document.querySelectorAll(".project-media video[autoplay]").forEach((video) => {
      video.removeAttribute("autoplay");
      video.pause();
    });
  }

  if (reducedMotion || !("IntersectionObserver" in window)) {
    revealItems.forEach((item) => item.classList.add("is-visible"));
  } else {
    const revealObserver = new IntersectionObserver(
      (entries, observer) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        });
      },
      { rootMargin: "0px 0px -8% 0px", threshold: 0 }
    );
    revealItems.forEach((item) => revealObserver.observe(item));
  }

  if ("IntersectionObserver" in window && sections.length) {
    const sectionObserver = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          navAnchors.forEach((anchor) => {
            const active = anchor.getAttribute("href") === `#${entry.target.id}`;
            anchor.classList.toggle("is-active", active);
            if (active) anchor.setAttribute("aria-current", "location");
            else anchor.removeAttribute("aria-current");
          });
        });
      },
      { rootMargin: "-35% 0px -55% 0px", threshold: 0 }
    );
    sections.forEach((section) => sectionObserver.observe(section));
  }

  const year = document.querySelector("[data-year]");
  if (year) year.textContent = new Date().getFullYear();
})();
