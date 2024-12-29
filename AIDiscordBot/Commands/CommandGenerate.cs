using Discord.WebSocket;
using DiscordMusicBot.Commands.Interfaces;
using Discord;
using DiscordMusicBot.Services.Interfaces;
using DiscordMusicBot.Services;
using System.Diagnostics;

namespace DiscordMusicBot.Commands.Commands
{
    internal class CommandGenerate : IDiscordCommand
    {
        private string _commandName = "generate";
        public string CommandName => _commandName;

        public SlashCommandBuilder Register()
        {
            return new SlashCommandBuilder()
                .WithName(_commandName)
                .WithDescription("Generate a image with the given prompt (Smaller resolution is faster!)")
                .AddOption(new SlashCommandOptionBuilder()
                    .WithName("prompt")
                    .WithDescription("Prompt used to generate the image.")
                    .WithType(ApplicationCommandOptionType.String)
                    .WithRequired(true))
                .AddOption(new SlashCommandOptionBuilder()
                    .WithName("width")
                    .WithDescription("Choose the width of the resolution.")
                    .WithType(ApplicationCommandOptionType.String) // Changed from Integer to String
                    .AddChoice("768", "768")                      // Both name and value are strings
                    .AddChoice("1024", "1024")
                    .AddChoice("1280", "1280")
                    .AddChoice("1440", "1440")
                    .WithRequired(true))
                .AddOption(new SlashCommandOptionBuilder()
                    .WithName("height")
                    .WithDescription("Choose the height of the resolution.")
                    .WithType(ApplicationCommandOptionType.String) // Changed from Integer to String
                    .AddChoice("768", "768")                      // Both name and value are strings
                    .AddChoice("1024", "1024")
                    .AddChoice("1280", "1280")
                    .AddChoice("1440", "1440")
                    .WithRequired(true))
                .AddOption(new SlashCommandOptionBuilder()
                    .WithName("helper_prompt")
                    .WithDescription("Helps with adding details to the image.")
                    .WithType(ApplicationCommandOptionType.String)
                    .AddChoice("Pixel art", "pixel art, ")
                    .AddChoice("Character Portrait mode", "absurdres, highly-detailed, best quality, masterpiece, very aesthetic, portrait, ")
                    .AddChoice("Landscape Wideshot mode", "absurdres, highly-detailed, best quality, masterpiece, very aesthetic, landscape, wide-shot, ")
                    .AddChoice("Raw (no help)", " ")
                    .WithRequired(true))
                .AddOption(new SlashCommandOptionBuilder()
                    .WithName("negative_prompt")
                    .WithDescription("Things the image should AVOID generating.")
                    .WithType(ApplicationCommandOptionType.String)
                    .AddChoice("Avoid low quality and bad hands", "lowres, worst quality, low quality, bad anatomy, bad hands, multiple views, ")
                    .AddChoice("Raw (no help)", " ")
                    .WithRequired(true));
        }

        public async Task ExecuteAsync(SocketSlashCommand command)
        {
            // Defer the response to allow more time for processing
            await command.DeferAsync();

            // Retrieve each option by name
            var promptOption = command.Data.Options.FirstOrDefault(option => option.Name == "prompt")?.Value as string;
            var widthOption = command.Data.Options.FirstOrDefault(option => option.Name == "width")?.Value as string;
            var heightOption = command.Data.Options.FirstOrDefault(option => option.Name == "height")?.Value as string;
            var helperPromptOption = command.Data.Options.FirstOrDefault(option => option.Name == "helper_prompt")?.Value as string;
            var negativePromptOption = command.Data.Options.FirstOrDefault(option => option.Name == "negative_prompt")?.Value as string;

            // Validate required options
            if (string.IsNullOrEmpty(promptOption) || string.IsNullOrEmpty(widthOption) || string.IsNullOrEmpty(heightOption) ||
                string.IsNullOrEmpty(helperPromptOption) || string.IsNullOrEmpty(negativePromptOption))
            {
                await command.FollowupAsync("One or more required options are missing.");
                return;
            }

            // Parse width and height to integers
            if (!int.TryParse(widthOption, out int width))
            {
                await command.FollowupAsync("Invalid width value.");
                return;
            }

            if (!int.TryParse(heightOption, out int height))
            {
                await command.FollowupAsync("Invalid height value.");
                return;
            }

            var response = await Service.Get<IServiceRequestManager>().SendRequestAsync(
                promptOption, width, height, 30, helperPromptOption);

            if (!string.IsNullOrEmpty(response))
            {
                //Hacky way to check if file was successfully generated. Else we print the error we're given.
                if (response.StartsWith("/") || response.StartsWith("c") || response.StartsWith("C"))
                {
                    await command.FollowupWithFileAsync(response);
                }
                else
                {
                    await command.FollowupAsync($"Error: {response}");
                }
            }
            else
            {
                await command.FollowupAsync("No response received from the image generation service.");
            }
        }
    }
}
