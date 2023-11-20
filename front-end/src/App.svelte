<script>
    import { tick } from "svelte";
    import SvelteMarkdown from "svelte-markdown";
    import botImage from "./assets/images/bot.png"; //update
    import meImage from "./assets/images/questionmark.png"; //update to user_avatar in the future
    import MdLink from "./lib/MdLink.svelte";


    let messages = [];
    let ragMode = true;
    let question = "Beispiel: 34-Jährige MS Patientin mit RRMS seit 3 Jahren, aktuell Neueinstellung mit Cladribin";
    let shouldAutoScroll = true;
    let input;
    let appState = "idle"; // or receiving
    let senderImages = { bot: botImage, me: meImage };
    let sessionId = '';
    let streamEndedNormally = false;
    let placeholderText = "Fangen Sie an zu tippen und finden Sie die besten Studien für Ihre PatientInnen.\n Beenden Sie Ihre Nachricht mit der Eingabetaste.";


    async function fetchSessionId() {
    try {
        const response = await fetch('http://localhost:8504/manage-session', { credentials: 'include' });
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        sessionId = data.session_id;
    } catch (error) {
        console.error("Failed to fetch session ID:", error);
    }
} 

async function resetSession() {
    try {
        const response = await fetch('http://localhost:8504/reset-session', {
            method: 'POST',
            credentials: 'include', // Important for including cookies
        });
        if (response.ok) {
            console.log('Session reset successfully');
            sessionId = ''; // Clear the session ID on the client side

            // Clear the messages to reset the conversation display
            messages = [];

            // Optionally, reset other states as needed
            // ...
        } else {
            console.error('Failed to reset session');
        }
    } catch (error) {
        console.error('Error resetting session:', error);
    }
}



async function send() {
    if (!question.trim().length) {
        return;
    }
    
    // Fetch the latest session ID before sending the message
    await fetchSessionId();
    
    appState = "receiving";
    addMessage("me", question, ragMode);
    const messageId = addMessage("bot", "", ragMode);
    
    try {
        const queryStreamUrl = `http://localhost:8504/query-stream?session_id=${sessionId}&text=${encodeURI(question)}&rag=true`;
        const evt = new EventSource(queryStreamUrl);
        question = "";

        evt.onmessage = (e) => {
            if (e.data) {
                const data = JSON.parse(e.data);
                if (data.init) {
                    updateMessage(messageId, "", data.model);
                    return;
                } else if (data.end) {
                    // Set flag when the final message is received
                    streamEndedNormally = true;
                    return;
                }
                updateMessage(messageId, data.token);
            }
        };
        
        evt.onerror = () => {
            if (!streamEndedNormally) {
                // Handle actual errors
                updateMessage(messageId, "Error: Stream closed unexpectedly.");
            } else {
                console.log("Stream closed normally.");
            }
            evt.close(); // Close the event source
        };
    } catch (e) {
        updateMessage(messageId, "Error: " + e.message);
    } finally {
        appState = "idle";
    }
}
    function updateMessage(existingId, text, model = null) {
        if (!existingId) {
            return;
        }
        const existingIdIndex = messages.findIndex((m) => m.id === existingId);
        if (existingIdIndex === -1) {
            return;
        }
        messages[existingIdIndex].text += text;
        if (model) {
            messages[existingIdIndex].model = model;
        }
        messages = messages;
    }


    function addMessage(from, text, rag) {
        const newId = Math.random().toString(36).substring(2, 9);
        const message = { id: newId, from, text, rag };
        messages = messages.concat([message]);
        return newId;
    }

    function scrollToBottom(node, _) {
        const scroll = () => node.scrollTo({ top: node.scrollHeight });
        scroll();
        return { update: () => shouldAutoScroll && scroll() };
    }

    function scrolling(e) {
        shouldAutoScroll = e.target.scrollTop + e.target.clientHeight > e.target.scrollHeight - 55;
    }

    $: appState === "idle" && input && focus(input);
    async function focus(node) {
        await tick();
        node.focus();
    }
    // send();
</script>

<main class="h-full text-sm bg-gradient-to-t from-indigo-100 bg-fixed overflow-hidden">
    {#if messages.length === 0}
    <!-- Placeholder text when no messages are present -->
    <div class="absolute top-0 left-0 right-0 bottom-0 flex items-center justify-center">
        <p class="text-center text-gray-600 text-lg opacity-50">
            {placeholderText}
        </p>
    </div>
{/if}
    <div on:scroll={scrolling} class="flex h-full flex-col py-12 overflow-y-auto" use:scrollToBottom={messages}>
        <div class="w-4/5 mx-auto flex flex-col mb-32">
            {#each messages as message (message.id)}
            <div
                class="max-w-[80%] min-w-[40%] rounded-lg p-4 mb-4 overflow-x-auto bg-white border border-indigo-200"
                class:self-end={message.from === "me"}
                class:text-right={message.from === "me"}
            >
                {#if message.from === "me"}
                    <!-- Flex container for "BenutzerIn" label and image for "me" messages -->
                    <div class="flex justify-end items-center gap-2">
                        <div class="text-sm font-bold">BenutzerIn</div>
                        <div class="w-12 h-12">
                            <img src={senderImages[message.from]} alt="" class="w-full h-full rounded-lg" />
                        </div>
                    </div>
                {/if}
                {#if message.from === "bot"}
                    <!-- Container for bot image and information remains unchanged -->
                    <div class="flex flex-row items-center gap-2">
                        <div class="w-12 h-12">
                            <img src={senderImages[message.from]} alt="" class="w-full h-full rounded-lg" />
                        </div>
                        <div class="text-sm">
                            <div><b>Clinical Study Bot</b></div>
                            <div>Model: {message.model ? message.model : ""}</div>
                        </div>
                    </div>
                {/if}
                <div class="mt-4"><SvelteMarkdown source={message.text} renderers={{ link: MdLink }} options={{
                    async: true,
                    pedantic: false,
                    gfm: true,
                  }} /></div>
            </div>
        {/each}        
        </div>
        <div class="text-sm w-full fixed bottom-16">
            <div class="shadow-lg bg-indigo-50 rounded-lg w-4/5 xl:w-2/3 2xl:w-1/2 mx-auto">
                <div class="rounded-t-lg px-4 py-2 font-light">
                    <!-- Reset Session Button -->
                    <button class="mt-2 bg-blue-500 hover:bg-blue-700 text-white font-bold py-1 px-4 rounded" on:click={resetSession}>
                        Neue Konversation starten
                    </button>
                </div>
                <form class="rounded-md w-full bg-white p-2 m-0" on:submit|preventDefault={send}>
                    <input
                        disabled={appState === "receiving"}
                        class="text-lg w-full bg-white focus:outline-none px-4"
                        bind:value={question}
                        bind:this={input}
                        type="text"
                    />
                </form>
            </div>
        </div>
    </div>
</main>



<style>
    :global(pre) {
        @apply bg-gray-100 rounded-lg p-4 border border-indigo-200;
    }
    :global(code) {
        @apply text-indigo-500;
    }
</style>
