% template and helper functions for 16-642 PS3 problem 2

rng(0); % initialize random number generator

b1 = [5,5]; % position of beacon 1
b2 = [15,5]; % position of beacon 2

% load pfData.mat
load pfData.mat

% define timestep for prediction
timestep = 0.1;

numSteps = size(q_groundTruth,2);
% initialize movie array
M(numSteps) = struct('cdata',[],'colormap',[]);

% Define number of particles in filter
num_particles = 2000;

% define placeholder for robot pose at every step
robot_pose = zeros(3,numSteps);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize particles
x_start = 0;
x_end = 20;

y_start = 0;
y_end = 10;

theta_start = 0;
theta_end = 2*pi;

x_rand = (x_end-x_start).*rand(num_particles,1) + x_start;
y_rand = (y_end-y_start).*rand(num_particles,1) + y_start;
theta_rand = (theta_end-theta_start).*rand(num_particles,1) + theta_start;

particles = [x_rand y_rand theta_rand];
particles = transpose(particles);

weights = ones(num_particles, 1)/num_particles;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% here is some code to plot the initial scene
figure(1)
plotParticles(particles); % particle cloud plotting helper function
hold on
plot([b1(1),b2(1)],[b1(2),b2(2)],'s',...
    'LineWidth',2,...
    'MarkerSize',10,...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor',[0.5,0.5,0.5]);
drawRobot(q_groundTruth(:,1), 'cyan'); % robot drawing helper function
axis equal
axis([0 20 0 10])
M(1) = getframe; % capture current view as movie frame

disp('hit return to continue')

% V and W
W = [0.6 0.3; 0.3 0.6];
V = [0.63 0.25; 0.25 0.63];

% beacon 1,2 locations in [x;y]
b1 = [5;5];
b2 = [15;5];

% iterate particle filter in this loop
for k = 2:numSteps

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %              put particle filter prediction step here               %
    
    for t1 = 1:num_particles
        v = mvnrnd([0;0], V);
        particles(1,t1) = particles(1,t1) + timestep.*(u(1,k) + v(1)).*cos(particles(3,t1)); 
        particles(2,t1) = particles(2,t1) + timestep.*(u(1,k) + v(1)).*sin(particles(3,t1));
        particles(3,t1) = particles(3,t1) + timestep.*((u(2,k) + v(2))); % eq. 8.37
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                put particle filter update step here                 
    % weight particles
    for t1 = 1:num_particles
        % get the beacon reading (groundtruth) (2x1 vector)
        current_dist = y(:,k-1);
        % get the output h(k) (2x1 vector)
        h = [ sqrt((particles(1,t1) - b1(1,1))^2 + (particles(2,t1) - b1(2,1))^2);
              sqrt((particles(1,t1) - b2(1,1))^2 + (particles(2,t1) - b2(2,1))^2) ];
        
        % knowing the current pos of particles and noisy sensor measurement
        % we can merge the distributions of the groundtruth 
        % and noisy measurement. We can merge multivariate pdfs using 
        % matlabs mvnpdf function
        weights(t1) = weights(t1) * mvnpdf(current_dist, h, W);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % resample particles

    % normalize the weights
    weights = weights/sum(weights);

    % make cumulative weight vector
    cumulative_sum = cumsum(weights);

    % make a dummy weight vector
    new_weights = zeros(size(weights));
    newparticles = zeros(size(particles));

    weight_size = size(weights);
  
    % resample
    for i = 1:weight_size
        
        % generate a random number between 0,1
        rand_num = rand();

        % find the element which is just lesser than rand_num
        [minval, idx] = min(abs(cumulative_sum - rand_num));
        newparticles(:, i) = particles(:, idx);
        new_weights(i) = 1/num_particles;
    end


    weights = ones(num_particles,1)/num_particles;
    particles = newparticles;

    % calculate the robot estimate from the particle cloud
    particle_estimate = mean(particles,2);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % plot particle cloud, robot, robot estimate, and robot trajectory here %
    % here is some code to plot the initial scene
    figure(1)
    clf
    plotParticles(particles); % particle cloud plotting helper function
    hold on
    plot([b1(1),b2(1)],[b1(2),b2(2)],'s',...
        'LineWidth',2,...
        'MarkerSize',10,...
        'MarkerEdgeColor','r',...
        'MarkerFaceColor',[0.5,0.5,0.5]);
    drawRobot(q_groundTruth(:,k), 'cyan'); % robot drawing helper function
    drawRobot(particle_estimate, 'cyan'); % robot drawing helper function
    plot(q_groundTruth(1,1:k), q_groundTruth(2,1:k))
    axis equal
    axis([0 20 0 10])
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % capture current figure and pause
    M(k) = getframe; % capture current view as movie frame
    pause(0.001)
    
end

% when you're ready, the following block of code will export the created 
% movie to an mp4 file
videoOut = VideoWriter('/home/sush/CMU/Assignment_Sem_1/MEC/Assignment_3/result_2000.mp4','Motion JPEG AVI');
videoOut.FrameRate=20;
open(videoOut);
for k=1:numSteps
  writeVideo(videoOut,M(k));
end
close(videoOut);

close

% helper function to plot a particle cloud
function plotParticles(particles)
plot(particles(1, :), particles(2, :), 'go')
line_length = 0.1;
quiver(particles(1, :), particles(2, :), line_length * cos(particles(3, :)), line_length * sin(particles(3, :)))
end


% helper function to plot a differential drive robot
function drawRobot(pose, color)
    
% draws a SE2 robot at pose
x = pose(1);
y = pose(2);
th = pose(3);

% define robot shape
robot = [-1 .5 1 .5 -1 -1;
          1  1 0 -1  -1 1 ];
tmp = size(robot);
numPts = tmp(2);
% scale robot if desired
scale = 0.5;
robot = robot*scale;

% convert pose into SE2 matrix
H = [ cos(th)   -sin(th)  x;
      sin(th)    cos(th)  y;
      0          0        1];

% create robot in position
robotPose = H*[robot; ones(1,numPts)];

% plot robot
plot(robotPose(1,:),robotPose(2,:),'k','LineWidth',2);
rFill = fill(robotPose(1,:),robotPose(2,:), color);
alpha(rFill,.2); % make fill semi transparent
end
