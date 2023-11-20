<script>
    export let href = "";
    export let title = "";
    export let text = "";

    // Regular expression to detect an email address
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

    // Determine if the href is an email
    const isEmail = href.startsWith('mailto:') || emailRegex.test(href);

    // Process the href to extract the email address if it's a mailto link
    const emailAddress = href.startsWith('mailto:') ? href.substring(7) : href;
</script>

{#if isEmail}
    <!-- Render email address as plain text -->
    <span>{text || emailAddress}</span>
{:else}
    <!-- For other links, render as a clickable hyperlink -->
    <a href={href} {title} target="_blank" rel="noreferrer noopener">{text}</a>
{/if}

<style>
    a {
        color: #007bff;
        text-decoration: none;
    }

    span {
        color: #007bff; /* Maintain the same color styling for email text */
    }
</style>
